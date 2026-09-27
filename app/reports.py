"""保管负荷报表（custody load report）：只读，不接入任何交接写流程。

查询语义
--------
GET /reports/custody-load 接收 holder / start / end 三个查询参数：

- 时间必须**携带时区**（显式偏移，如 ``2026-09-01T00:00:00+08:00`` 或
  ``...Z``），服务端统一转换为 UTC；朴素时间（无时区）一律 400
  invalid_time_range；
- start 必须严格早于 end，end 不得晚于查询时刻（UTC now），否则 400
  invalid_time_range；
- 统计窗口为半开区间 ``[start, end)``。

快照一致性
----------
与审计检索 / 清单核对相同：在**同一个 SQLite 读事务**内先把「当时最大的事件
数据库 id」固定为快照上界 snapshot_event_id（全库无事件时为 0），随后只使用
``id <= snapshot_event_id`` 的不可变事件集合重建——重建期间新追加的移交不会
进入本次报表。

全库持有区间重建
----------------
仅用快照内事件重建该持有人（holder）在**全库全部链**上的持有区间：

- 创世（seq=0）由 ``chains.first_holder`` 开始持有；
- 移交（seq>=1）以 ``events.receiver`` 接续持有，移交的 giver 不作为重建
  依据（它应等于链上前一任持有人，仅以 receiver 接续即可）；
- 每条链的事件按 **(created_at, 事件数据库 id)** 升序处理（链内该顺序与 seq
  一致，id 用于跨表的稳定决胜）；
- **先还原窗口前状态**：先顺序应用 ``created_at < start`` 的全部事件（包括
  创世与移交），得到窗口起点时刻该人是否持有每条链；再应用窗口内事件，并把
  得到的各链持有区间统一裁剪到 ``[start, end)``；
- **同一时刻的变化合并后计算**：同一链上 created_at 相同的事件先合并为一个
  时刻，仅以该时刻最后一条事件（id 最大）的 receiver 作为该时刻之后的持有人，
  因此「同一秒交出又收回 / 收回又交出」的中间态绝不产生负荷；
- **每条链最多计一次**：一条链在任意时刻对该人贡献的持有链数只能是 0 或 1；
- **自转交不增负荷**：receiver 与当前持有人相同（自己交给自己）时持有关系
  不变，既不结束也不重新开始区间，自然不增加负荷。

报表输出
--------
- ``max_concurrent_chains``：窗口内该人**同时持有链数**的最大值（可为 0）；
- ``peak_intervals``：达到**正峰值**（>0 的最大值）的全部**合并连续区间**：
  相邻且持有链数相同的峰值时段在中间事件导致的零长度跳变后仍合并为一个区间；
- ``held_chain_seconds``：持有链数对时间的积分（链·秒），按
  ``Σ 区间内持有链数 × 持续秒数`` 计算。
- 该人在快照内没有任何持有记录（无创世、从未作为 receiver 出现）时：
  数值为 0、``peak_intervals`` 为空列表。

本模块**只读**：不追加事件、不推进链尾、不写任何表，也不改变邀约 / 封存等
任何交接流程。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from .database import read_tx
from .service import ChainError

# 积分（链·秒）的小数输出精度：所有事件时刻均为整秒，正常结果必为整数；
# 仅当调用方提交带亚秒的 start/end 时才可能出现小数，统一保留 6 位并消除
# 二进制浮点尾差，整数结果以 int 返回。
_INTEGRAL_DECIMALS = 6


def parse_zoned_time(value: str, field: str) -> datetime:
    """解析查询参数中的 ISO-8601 时间，**要求显式时区**并转换为 UTC。

    与审计检索（朴素时间按 UTC 解释）不同，报表接口按需求强制要求携带时区：
    无法解析或得到朴素 datetime（tzinfo 为 None）一律 400 invalid_time_range。
    """
    text = value.strip()
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ChainError(
            400,
            "invalid_time_range",
            f"{field} 不是合法的 ISO-8601 时间（须携带时区，如 "
            f"2026-09-01T00:00:00+08:00 或 2026-09-01T00:00:00Z）: {text}",
        ) from exc
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ChainError(
            400,
            "invalid_time_range",
            f"{field} 必须携带时区偏移（如 +08:00 / Z），不接受无时区的朴素时间: {text}",
        )
    return dt.astimezone(timezone.utc)


def _validate_window(
    start_raw: str,
    end_raw: str,
    now: Optional[datetime] = None,
) -> tuple[datetime, datetime]:
    """校验时间窗口：显式时区 -> UTC；start < end；end 不晚于查询时刻。"""
    start = parse_zoned_time(start_raw, "start")
    end = parse_zoned_time(end_raw, "end")
    if not start < end:
        raise ChainError(
            400,
            "invalid_time_range",
            "start 必须严格早于 end（统计窗口为半开区间 [start, end)）",
        )
    now = now or datetime.now(timezone.utc)
    if end > now:
        raise ChainError(
            400,
            "invalid_time_range",
            "end 不得晚于查询时刻（UTC）：报表只统计已经发生的持有区间，"
            f"end={end.isoformat()} 晚于当前时刻 {now.isoformat()}",
        )
    return start, end


def _reconstruct_chain_intervals(
    rows: list[Any],
    holder: str,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, datetime]]:
    """用单条链的全部事件重建该链对 holder 的持有区间并裁剪到 [start, end)。

    rows 为该链快照内的事件行（至少含创世事件），调用方已按
    (created_at, id) 升序排序。事件按 created_at 分组合并：同一时刻的多个事件
    只保留该时刻最后一条（id 最大）的语义——创世时刻以 first_holder 为持有人；
    移交时刻以最后一条事件的 receiver 接续。自转交（receiver 与当前持有人
    相同）不改变持有关系。

    处理方式天然「先还原窗口前状态」：< start 的事件照常顺序应用，只是其产生
    的区间在最后统一与 [start, end) 求交。
    """
    current: Optional[str] = None
    hold_since: Optional[datetime] = None
    intervals: list[tuple[datetime, datetime]] = []

    index = 0
    count = len(rows)
    while index < count:
        group_ts = rows[index]["created_at"]
        ts = _parse_event_time(group_ts)
        # 合并同一链上同一时刻的全部事件：以 id 最大（rows 已按 (created_at,id)
        # 升序）的事件决定该时刻之后的持有人，中间态一律丢弃。
        last = rows[index]
        while index + 1 < count and rows[index + 1]["created_at"] == group_ts:
            index += 1
            last = rows[index]
        index += 1

        # 该时刻之后的持有人由分组的最后一条事件（id 最大）决定：最后一条是
        # 创世（此组仅含创世）时取 first_holder；否则取最后一条移交的 receiver
        # 接续——创世后同刻若有移交，最后一条必是移交，由 receiver 决胜。
        if last["seq"] == 0:
            successor = last["first_holder"]
        else:
            successor = last["receiver"]
        new_holder = holder if successor == holder else None

        # 自转交（new_holder 与当前持有人相同）：什么都不变，
        # 既不结束也不重新开始持有区间。
        if new_holder is not None and new_holder == current:
            continue

        if new_holder is not None and current is None:
            # 开始持有（创世于 first_holder，或某次移交由该人接收）。
            current = new_holder
            hold_since = ts
        elif new_holder is None and current is not None:
            # 结束持有（某次移交给了别人）。
            intervals.append((hold_since, ts))
            current = None
            hold_since = None
        # current 为 None 且 new_holder 为 None：该人从未持有，忽略。

    # 窗口末端仍持有：区间以窗口上界 end 收口（实际持有可能延续到查询之后，
    # 但 end 已被限制为不晚于查询时刻，裁剪后至多计到 end）。
    if current is not None and hold_since is not None:
        intervals.append((hold_since, end))

    return [
        (max(lo, start), min(hi, end))
        for lo, hi in intervals
        if max(lo, start) < min(hi, end)
    ]


def _parse_event_time(value: str) -> datetime:
    """事件落库时间解析为 UTC datetime（落库统一为显式 +00:00 ISO-8601）。"""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _aggregate(
    intervals_by_chain: list[list[tuple[datetime, datetime]]],
) -> dict[str, Any]:
    """把每条链（0/1 持有）的区间集合汇总为峰值、峰值区间与积分。

    扫描线：在每个区间端点记录 +1 / -1 增量，按时刻排序后逐段累加，得到
    「自某时刻起至下一变化时刻止」的恒定持有链数。每条链各自的区间互不重叠，
    因而同一链在任一时段至多贡献 1。同一时刻的全部增减在该时间点一次性轧差，
    该时刻不产生零长度分段——因此「同时刻先减后增、净值不变」的跳变不会打断
    峰值区间的连续性。
    """
    deltas: dict[datetime, int] = defaultdict(int)
    for intervals in intervals_by_chain:
        for lo, hi in intervals:
            deltas[lo] += 1
            deltas[hi] -= 1

    if not deltas:
        return {
            "max_concurrent_chains": 0,
            "peak_intervals": [],
            "held_chain_seconds": 0,
        }

    # 恒定持有链数分段：仅保留正长度段（lo < hi）；同一时刻的多个增量在进入
    # 下一不同时刻前已轧差，中间值不外显。
    segments: list[tuple[datetime, datetime, int]] = []
    current_count = 0
    prev_ts: Optional[datetime] = None
    for ts, delta in sorted(deltas.items()):
        if prev_ts is not None and ts > prev_ts:
            segments.append((prev_ts, ts, current_count))
        current_count += delta
        prev_ts = ts

    peak = max((c for _, _, c in segments), default=0)
    integral = sum(
        c * (hi - lo).total_seconds() for lo, hi, c in segments
    )

    if peak <= 0:
        return {
            "max_concurrent_chains": 0,
            "peak_intervals": [],
            "held_chain_seconds": _normalize_seconds(integral),
        }

    # 达到正峰值的全部连续区间：相邻峰值段端点相接即合并（端点相同意味着中间
    # 只有同时刻的零长度跳变，已在分段阶段消除）；中间夹有任何正长度的非峰值
    # 段（含持有链数为 0）则断开为新区间。
    peak_intervals: list[dict[str, str]] = []
    for lo, hi, c in segments:
        if c != peak:
            continue
        if peak_intervals and peak_intervals[-1]["end"] == lo.isoformat():
            peak_intervals[-1]["end"] = hi.isoformat()
        else:
            peak_intervals.append(
                {"start": lo.isoformat(), "end": hi.isoformat()}
            )

    return {
        "max_concurrent_chains": peak,
        "peak_intervals": peak_intervals,
        "held_chain_seconds": _normalize_seconds(integral),
    }


def _normalize_seconds(value: float) -> Any:
    """积分输出：整秒为 int；带亚秒窗口时四舍五入到 6 位小数。"""
    rounded = round(value, _INTEGRAL_DECIMALS)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def custody_load_report(
    holder: str,
    start: str,
    end: str,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """生成 holder 在 [start, end) 内的保管负荷报表（只读，单一读事务快照）。

    在同一个 SQLite 读事务内固定 snapshot_event_id（MAX(events.id)，空库为
    0），并一次性取出「创世 first_holder 为该人，或快照内曾有移交以该人为
    receiver」的全部链及其事件（INNER JOIN 保证链存在且至少有创世事件），
    随后在内存中逐链重建、裁剪，再跨链汇总。该人在快照内没有任何持有记录时
    数值为 0、峰值区间为空。
    """
    holder = (holder or "").strip()
    if not holder:
        raise ChainError(400, "invalid_holder", "holder 不能为空")
    start_dt, end_dt = _validate_window(start, end, now)

    with read_tx() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM events"
        ).fetchone()
        snapshot_event_id = row["m"]

        rows = conn.execute(
            """SELECT e.id, e.chain_id, e.seq, e.receiver, e.created_at,
                      c.first_holder
               FROM events e
               INNER JOIN chains c ON c.chain_id = e.chain_id
               WHERE e.id <= ?
                 AND e.chain_id IN (
                     SELECT ch.chain_id
                     FROM chains ch
                     WHERE ch.first_holder = ?
                     UNION
                     SELECT ev.chain_id
                     FROM events ev
                     WHERE ev.id <= ?
                       AND ev.seq > 0
                       AND ev.receiver = ?
                 )
               ORDER BY e.chain_id ASC, e.created_at ASC, e.id ASC""",
            (snapshot_event_id, holder, snapshot_event_id, holder),
        ).fetchall()

    # 按链分组（SQL 已按 chain_id, created_at, id 排序，直接切分即可）。
    intervals_by_chain: list[list[tuple[datetime, datetime]]] = []
    current_chain: Optional[str] = None
    chain_rows: list[Any] = []
    for r in rows:
        if current_chain is None or r["chain_id"] == current_chain:
            chain_rows.append(r)
        else:
            intervals_by_chain.append(
                _reconstruct_chain_intervals(chain_rows, holder, start_dt, end_dt)
            )
            chain_rows = [r]
        current_chain = r["chain_id"]
    if chain_rows:
        intervals_by_chain.append(
            _reconstruct_chain_intervals(chain_rows, holder, start_dt, end_dt)
        )

    result = _aggregate(intervals_by_chain)
    return {
        "holder": holder,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "snapshot_event_id": snapshot_event_id,
        **result,
    }
