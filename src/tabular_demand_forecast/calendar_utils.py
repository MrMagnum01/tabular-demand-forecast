"""Weekly calendar helpers, derived from eval/protocol.json's calendar block."""
from __future__ import annotations

import datetime as dt

from .protocol import load_protocol


def start_date() -> dt.date:
    return dt.date.fromisoformat(load_protocol()["calendar"]["start_date"])


def n_weeks() -> int:
    return int(load_protocol()["calendar"]["n_weeks"])


def week_start_date(week_index: int) -> dt.date:
    """week_start_date(w) = start_date + 7*(w-1) days, per protocol.calendar.date_formula."""
    return start_date() + dt.timedelta(days=7 * (week_index - 1))


def week_of_year(week_index: int) -> int:
    """ISO calendar week-of-year (1-53) of week_start_date(week_index)."""
    return week_start_date(week_index).isocalendar()[1]


def origin(week_index: int) -> dt.date:
    """origin(t): the instant target week t begins; equals week_start_date(t)."""
    return week_start_date(week_index)
