"""The earnings book's page starts its equity curve at the close before it took the account over."""

from site_common import since_history


def _rows(*dates):
    return [{"date": d, "equity": 100_000.0 + i, "profit_loss": 0.0} for i, d in enumerate(dates)]


def test_keeps_the_last_close_before_the_start_as_the_starting_equity():
    history = _rows("2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26")
    kept = since_history(history, "2026-10-23")
    assert [r["date"] for r in kept] == ["2026-10-22", "2026-10-23", "2026-10-26"]


def test_one_session_of_its_own_still_makes_two_points():
    # The page draws nothing from a single point; the takeover close makes the second.
    history = _rows("2026-10-21", "2026-10-22", "2026-10-23")
    assert [r["date"] for r in since_history(history, "2026-10-23")] == ["2026-10-22", "2026-10-23"]


def test_before_the_first_bar_is_posted_only_the_start_remains():
    history = _rows("2026-10-21", "2026-10-22")
    assert [r["date"] for r in since_history(history, "2026-10-23")] == ["2026-10-22"]


def test_a_book_without_earlier_history_is_unchanged():
    history = _rows("2026-10-23", "2026-10-26")
    assert since_history(history, "2026-10-23") == history
