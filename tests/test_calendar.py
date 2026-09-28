import datetime as dt

from tabular_demand_forecast.calendar_utils import week_of_year, week_start_date


def test_week_start_date_matches_protocol_worked_examples():
    assert week_start_date(1) == dt.date(2022, 1, 3)
    assert week_start_date(52) == dt.date(2022, 12, 26)
    assert week_start_date(53) == dt.date(2023, 1, 2)
    assert week_start_date(78) == dt.date(2023, 6, 26)
    assert week_start_date(79) == dt.date(2023, 7, 3)
    assert week_start_date(91) == dt.date(2023, 9, 25)
    assert week_start_date(92) == dt.date(2023, 10, 2)
    assert week_start_date(104) == dt.date(2023, 12, 25)


def test_week_1_is_a_monday():
    assert week_start_date(1).isoweekday() == 1


def test_week_of_year_iso():
    assert week_of_year(1) == 1
    assert week_of_year(53) == 1  # 2023-01-02 is ISO week 2023-W01
