"""Bot local time. Everything is stored in unix seconds; days and windows use persona.timezone."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


def parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def local_dt(ts: int, tz: str) -> datetime:
    return datetime.fromtimestamp(ts, ZoneInfo(tz))


def local_date(ts: int, tz: str) -> date:
    return local_dt(ts, tz).date()


def day_start(ts: int, tz: str) -> int:
    """Unix time of local midnight of the day that contains ts.

    Needed by "per day" counters that are computed straight from the tables
    (messages/bot_replies) instead of a state key with a date suffix (``day_key``),
    such as the presence cap.
    """
    zone = ZoneInfo(tz)
    return int(datetime.combine(local_date(ts, tz), time(0, 0), tzinfo=zone).timestamp())


def day_key(prefix: str, ts: int, tz: str) -> str:
    """State key with a daily suffix: 'ambient_count:2026-09-10'."""
    return f"{prefix}:{local_date(ts, tz).isoformat()}"


def week_key(prefix: str, ts: int, tz: str) -> str:
    """State key with an ISO weekly suffix: 'spontaneous_count:2026-W37'."""
    year, week, _ = local_date(ts, tz).isocalendar()
    return f"{prefix}:{year}-W{week:02d}"


def in_window(ts: int, tz: str, window: tuple[str, str]) -> bool:
    """Whether the moment falls into the ["HH:MM", "HH:MM") window in local time.

    Start is inclusive, end is exclusive. A window across midnight ("23:00", "06:00")
    is supported. An empty window (start == end) is always False.
    """
    start, end = parse_hhmm(window[0]), parse_hhmm(window[1])
    now = local_dt(ts, tz).time().replace(tzinfo=None)
    if start == end:
        return False
    if start < end:
        return start <= now < end
    return now >= start or now < end


def seconds_until(ts: int, tz: str, hhmm: str) -> int:
    """Seconds from ts to the next occurrence of hhmm in local time.

    If hhmm has already passed today, count to tomorrow's. Exactly now is 0.
    """
    zone = ZoneInfo(tz)
    now = local_dt(ts, tz)
    target = datetime.combine(now.date(), parse_hhmm(hhmm), tzinfo=zone)
    if target < now:
        target = datetime.combine(now.date() + timedelta(days=1), parse_hhmm(hhmm), tzinfo=zone)
    return int((target - now).total_seconds())
