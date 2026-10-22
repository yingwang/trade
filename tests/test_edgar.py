"""The EDGAR point-in-time loader on hand-built facts (offline) and, when the
sample is on disk, on Apple's real company-facts file."""

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from quant.data import edgar as E

SAMPLE_DIR = "data/edgar_cache/edgar"


def _fact(start, end, val, filed, form="10-Q", accn=None):
    row = {"end": end, "val": val, "accn": accn or f"acc-{filed}", "fy": 2024, "fp": "FY", "form": form, "filed": filed}
    if start is not None:
        row["start"] = start
    return row


def _facts(**concepts):
    """{concept: [rows]} (us-gaap, USD unless the name says otherwise) -> company-facts JSON."""
    out = {"cik": 1, "entityName": "Test Co", "facts": {"us-gaap": {}, "dei": {}}}
    for name, rows in concepts.items():
        taxonomy, concept = ("dei", name[4:]) if name.startswith("dei_") else ("us-gaap", name)
        unit = "USD/shares" if "PerShare" in concept else "shares" if "Shares" in concept else "USD"
        out["facts"][taxonomy][concept] = {"label": concept, "units": {unit: rows}}
    return out


# A calendar-year company: three 10-Qs, a 10-K whose year implies Q4, then a
# 10-Q that restates Q1 of the previous year in its comparative column.
INCOME = [
    _fact("2023-01-01", "2023-03-31", 100, "2023-05-01"),
    _fact("2023-04-01", "2023-06-30", 110, "2023-08-01"),
    _fact("2023-01-01", "2023-06-30", 210, "2023-08-01"),
    _fact("2023-07-01", "2023-09-30", 120, "2023-11-01"),
    _fact("2023-01-01", "2023-09-30", 330, "2023-11-01"),
    _fact("2023-01-01", "2023-12-31", 460, "2024-02-15", form="10-K"),
    _fact("2024-01-01", "2024-03-31", 140, "2024-05-01"),
    _fact("2023-01-01", "2023-03-31", 105, "2024-05-01"),  # restated comparative
]
ASSETS = [
    _fact(None, "2023-03-31", 1000, "2023-05-01"),
    _fact(None, "2023-06-30", 1050, "2023-08-01"),
    _fact(None, "2023-09-30", 1100, "2023-11-01"),
    _fact(None, "2023-12-31", 1200, "2024-02-15", form="10-K"),
    _fact(None, "2024-03-31", 1300, "2024-05-01"),
    _fact(None, "2023-03-31", 1010, "2024-05-01"),  # restated comparative
]
SUBMISSIONS = {
    "cik": "0000000001",
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q", "8-K", "8-K/A", "8-K"],
            "filingDate": ["2024-04-25", "2024-05-01", "2024-03-01", "2024-01-30", "2024-01-29"],
            "items": ["2.02,9.01", "", "5.02", "2.02", "2.02,7.01"],
            "accessionNumber": ["a", "b", "c", "d", "e"],
        },
        "files": [],
    },
}


def test_extract_facts_merges_candidates_in_priority_order():
    facts = _facts(
        Revenues=[_fact("2023-01-01", "2023-03-31", 500, "2023-05-01"),
                  _fact("2023-01-01", "2023-03-31", 500, "2023-05-01", form="8-K", accn="dup")],
        SalesRevenueNet=[_fact("2023-01-01", "2023-03-31", 499, "2023-05-01"),
                         _fact("2022-01-01", "2022-03-31", 450, "2022-05-01"),
                         _fact("2102-01-01", "2102-03-31", 450, "2022-05-01")],
        Assets=[_fact(None, "2023-03-31", 1000, "2023-05-01")],
    )
    rows = E.extract_facts(facts, ["Revenues", "SalesRevenueNet"])
    assert list(rows.columns) == ["start", "end", "val", "filed", "form", "accn", "concept"]
    # The 2023 quarter comes from Revenues (higher priority) and only once;
    # the 2022 quarter exists only under SalesRevenueNet and is kept.
    assert len(rows) == 2
    assert rows.iloc[0]["concept"] == "SalesRevenueNet" and rows.iloc[0]["val"] == 450
    assert rows.iloc[1]["concept"] == "Revenues" and rows.iloc[1]["val"] == 500
    assert rows["filed"].is_monotonic_increasing
    assert str(rows["filed"].dtype).startswith("datetime64")
    # A period ending long after its filing date is a tagging error, dropped.
    assert not (rows["end"] > pd.Timestamp("2100-01-01")).any()
    # Instants have no start; a missing concept gives an empty frame.
    inst = E.extract_facts(facts, ["Assets"])
    assert inst["start"].isna().all() and len(inst) == 1
    assert E.extract_facts(facts, ["GrossProfit"]).empty
    assert E.extract_facts(facts, ["Revenues"], units=("EUR",)).empty


def test_discrete_quarters_reports_and_derives_q4_per_filing():
    q = E.discrete_quarters(E.extract_facts(_facts(NetIncomeLoss=INCOME), ["NetIncomeLoss"]))
    assert list(q.columns) == ["end", "start", "val", "filed", "derived"]
    reported = q[~q["derived"]]
    assert sorted(reported["val"]) == [100, 105, 110, 120, 140]
    # Q4 2023 from the 10-K: the year minus what was accumulated by the
    # previous quarter end, dated by the 10-K, starting at that quarter end.
    q4 = q[q["derived"] & (q["end"] == "2023-12-31")]
    assert len(q4) == 1
    first = q4.iloc[0]
    assert first["val"] == 460 - 100 - 110 - 120 and first["filed"] == pd.Timestamp("2024-02-15")
    assert first["start"] == pd.Timestamp("2023-09-30")
    # Half-year and nine-month cumulatives reproduce the reported Q2/Q3 and
    # are dropped as duplicates of the same filing; the restated Q1 alone
    # (filed 2024-05-01) does not re-derive anything.
    assert (q[q["end"].isin(["2023-06-30", "2023-09-30"])]["derived"] == False).all()  # noqa: E712
    assert not (q["filed"] == "2024-05-01").sum() > 2
    # A restated year (10-K/A) yields a second Q4 version on its own filing date.
    amended = INCOME + [_fact("2023-01-01", "2023-12-31", 465, "2024-06-01", form="10-K/A")]
    q = E.discrete_quarters(E.extract_facts(_facts(NetIncomeLoss=amended), ["NetIncomeLoss"]))
    q4 = q[q["derived"] & (q["end"] == "2023-12-31")].sort_values("filed")
    assert q4["val"].tolist() == [130, 135] and q4["filed"].iloc[1] == pd.Timestamp("2024-06-01")
    assert E.discrete_quarters(E.extract_facts(_facts(), ["NetIncomeLoss"])).empty


def test_discrete_quarters_from_year_to_date_only_and_nine_month_fallback():
    # Cash-flow style: only cumulative periods are tagged.
    cfo = [
        _fact("2023-01-01", "2023-03-31", 10, "2023-05-01"),
        _fact("2023-01-01", "2023-06-30", 25, "2023-08-01"),
        _fact("2023-01-01", "2023-09-30", 45, "2023-11-01"),
        _fact("2023-01-01", "2023-12-31", 70, "2024-02-15", form="10-K"),
    ]
    q = E.discrete_quarters(E.extract_facts(_facts(NetCashProvidedByUsedInOperatingActivities=cfo),
                                            ["NetCashProvidedByUsedInOperatingActivities"]))
    by_end = q.set_index("end")["val"]
    assert by_end.to_dict() == {pd.Timestamp("2023-03-31"): 10, pd.Timestamp("2023-06-30"): 15,
                                pd.Timestamp("2023-09-30"): 20, pd.Timestamp("2023-12-31"): 25}
    assert q.set_index("end")["derived"].to_dict() == {pd.Timestamp("2023-03-31"): False, pd.Timestamp("2023-06-30"): True,
                                                       pd.Timestamp("2023-09-30"): True, pd.Timestamp("2023-12-31"): True}
    # Without the nine-month figure Q4 cannot be derived when Q2 is also
    # missing; with it, Q4 = year minus nine months even then.
    sparse = [
        _fact("2023-01-01", "2023-03-31", 10, "2023-05-01"),
        _fact("2023-07-01", "2023-09-30", 20, "2023-11-01"),
        _fact("2023-01-01", "2023-12-31", 70, "2024-02-15", form="10-K"),
    ]
    q = E.discrete_quarters(E.extract_facts(_facts(NetIncomeLoss=sparse), ["NetIncomeLoss"]))
    assert "2023-12-31" not in set(q["end"].dt.strftime("%Y-%m-%d"))
    q = E.discrete_quarters(E.extract_facts(_facts(NetIncomeLoss=sparse + [_fact("2023-01-01", "2023-09-30", 45, "2023-11-01")]),
                                            ["NetIncomeLoss"]))
    q4 = q[q["end"] == "2023-12-31"]
    assert len(q4) == 1 and q4.iloc[0]["val"] == 25 and q4.iloc[0]["derived"]


def test_annual_durations_and_instants():
    rows = E.extract_facts(_facts(NetIncomeLoss=INCOME), ["NetIncomeLoss"])
    year = E.annual_durations(rows)
    assert len(year) == 1 and year.iloc[0]["val"] == 460
    inst = E.instants(E.extract_facts(_facts(Assets=ASSETS), ["Assets"]))
    assert list(inst.columns) == ["end", "val", "filed"] and len(inst) == 6
    assert E.instants(rows).empty and E.annual_durations(E.instants(rows)).empty


def test_as_of_knows_nothing_before_the_filing_and_uses_restatements_after():
    q = E.discrete_quarters(E.extract_facts(_facts(NetIncomeLoss=INCOME), ["NetIncomeLoss"]))
    dates = pd.DatetimeIndex(["2023-04-30", "2023-05-01", "2023-08-01", "2024-02-14", "2024-02-15", "2024-05-01", "2025-06-30"])
    s = E.as_of(q, dates)
    assert np.isnan(s.iloc[0])                       # Q1 is filed on 2023-05-01, unknown the day before
    assert s.loc["2023-05-01"] == 100
    assert s.loc["2023-08-01"] == 110                # the latest period end wins
    assert s.loc["2024-02-14"] == 120 and s.loc["2024-02-15"] == 130
    assert s.loc["2024-05-01"] == 140
    # Trailing sums: at the 10-K the year itself; a quarter later the new
    # quarter replaces the one that dropped out (the restated Q1 is outside
    # the window and the derived Q4 keeps the 10-K's vintage).
    t = E.as_of(E.ttm(q), dates)
    assert t.loc["2024-02-15"] == 460
    assert t.loc["2024-05-01"] == 140 + 130 + 120 + 110
    assert np.isnan(t.loc["2024-02-14"])
    # Staleness: the last period (2024-03-31) is more than 400 days old by mid-2025.
    assert s.loc["2025-06-30"] == 140
    assert np.isnan(E.as_of(q, dates, stale_after_days=400).loc["2025-06-30"])
    assert E.as_of(q.iloc[:0], dates).isna().all()


def _naive_as_of(events, date):
    sub = events[events["filed"] <= date]
    if sub.empty:
        return np.nan
    sub = sub[sub["end"] == sub["end"].max()]
    return sub.sort_values("filed", kind="stable").iloc[-1]["val"]


def test_as_of_vectorised_equals_naive_on_random_events():
    rng = np.random.default_rng(7)
    ends = pd.date_range("2015-03-31", periods=30, freq="QE")
    rows = []
    for end in ends:
        for version in range(rng.integers(1, 4)):
            filed = end + pd.Timedelta(days=int(rng.integers(20, 400)))
            rows.append({"end": end, "val": float(rng.normal()), "filed": filed})
    events = pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(rng.choice(pd.date_range("2015-01-01", "2023-12-31"), size=300, replace=False)))
    fast = E.as_of(events, dates)
    naive = pd.Series([_naive_as_of(events, d) for d in dates], index=dates)
    pd.testing.assert_series_equal(fast, naive, check_names=False)


def test_ttm_needs_four_consecutive_quarters():
    rows = [_fact(f"{y}-{m:02d}-01", f"{y}-{m + 2:02d}-{'30' if m + 2 in (6, 9) else '31'}", 10 * i + 1,
                  f"{y}-{m + 3:02d}-15" if m < 10 else f"{y + 1}-01-15")
            for i, (y, m) in enumerate([(2022, 1), (2022, 4), (2022, 7), (2022, 10), (2023, 1), (2023, 4), (2023, 10)])]
    q = E.discrete_quarters(E.extract_facts(_facts(NetIncomeLoss=rows), ["NetIncomeLoss"]))
    t = E.ttm(q)
    assert list(t.columns) == ["end", "val", "filed"]
    # Four quarters exist from 2022-12-31 (sum 1+11+21+31) and 2023-03-31;
    # 2023-06-30 is the fourth of a chain too; Q3 2023 is missing, so the
    # quarter ending 2023-12-31 has no trailing year.
    by_end = t.set_index("end")["val"]
    assert by_end.loc["2022-12-31"] == 64 and by_end.loc["2023-03-31"] == 104 and by_end.loc["2023-06-30"] == 144
    assert pd.Timestamp("2023-12-31") not in by_end.index
    assert E.ttm(q.iloc[:0]).empty


def test_seasonal_sue_standardises_the_year_over_year_change():
    diffs = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3]
    eps = [1.0, 1.0, 1.0, 1.0]
    for d in diffs:
        eps.append(eps[-4] + d)
    ends = pd.date_range("2020-03-31", periods=len(eps), freq="QE")
    rows = [_fact((end - pd.Timedelta(days=89)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), v,
                  (end + pd.Timedelta(days=30)).strftime("%Y-%m-%d"))
            for end, v in zip(ends, eps)]
    q = E.discrete_quarters(E.extract_facts(_facts(EarningsPerShareDiluted=rows), ["EarningsPerShareDiluted"], units=("USD/shares",)))
    sue = E.seasonal_sue(q, lookback=8)
    last = sue[sue["end"] == ends[-1]]
    assert len(last) == 1
    expected = 3 / np.std([1, 2, 1, 2, 1, 2, 1, 2], ddof=1)
    assert last.iloc[0]["val"] == pytest.approx(expected)
    assert last.iloc[0]["filed"] == ends[-1] + pd.Timedelta(days=30)
    # Needs six prior differences: the first SUE is for the 11th quarter
    # (four for the lag, six differences after that).
    assert sue["end"].min() == ends[10]
    assert E.seasonal_sue(q.iloc[:0]).empty


def test_lagged_gives_the_value_four_quarters_earlier_as_of_each_filing():
    inst = E.instants(E.extract_facts(_facts(Assets=ASSETS), ["Assets"]))
    lag = E.lagged(inst, quarters_back=4)
    # 2024-03-31 looks back to 2023-03-31: 1000 as filed originally with the
    # 2024-05-01 10-Q, but that same 10-Q restated it to 1010.
    rows = lag[lag["end"] == "2024-03-31"].sort_values("filed")
    assert rows["val"].tolist() == [1010] and rows["filed"].tolist() == [pd.Timestamp("2024-05-01")]
    assert set(lag["end"]) == {pd.Timestamp("2024-03-31")}
    # Restated later than the current period's own filing: a new event.
    later = pd.concat([inst, pd.DataFrame({"end": [pd.Timestamp("2023-03-31")], "val": [1020.0], "filed": [pd.Timestamp("2024-08-01")]})])
    lag = E.lagged(later).sort_values("filed")
    assert lag["val"].tolist() == [1010, 1020] and lag["filed"].iloc[-1] == pd.Timestamp("2024-08-01")
    assert E.lagged(inst.iloc[:0]).empty


def test_split_adjusted_rescales_by_splits_after_the_filing_date():
    events = pd.DataFrame({"end": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30"]),
                           "val": [100.0, 100.0, 400.0],
                           "filed": pd.to_datetime(["2023-05-01", "2023-08-01", "2023-11-01"])})
    splits = pd.Series({pd.Timestamp("2023-09-15"): 4.0, pd.Timestamp("2010-01-01"): 2.0})
    shares = E.split_adjusted(events, splits)
    assert shares["val"].tolist() == [400.0, 400.0, 400.0]
    eps = E.split_adjusted(events, splits, per_share=True)
    assert eps["val"].tolist() == [25.0, 25.0, 400.0]
    assert E.split_adjusted(events, None) is events
    assert E.split_adjusted(events, pd.Series(dtype=float))["val"].tolist() == [100.0, 100.0, 400.0]


def test_earnings_release_dates_are_the_item_2_02_eight_ks():
    dates = E.earnings_release_dates(SUBMISSIONS)
    assert isinstance(dates, pd.DatetimeIndex)
    assert [d.strftime("%Y-%m-%d") for d in dates] == ["2024-01-29", "2024-01-30", "2024-04-25"]
    assert len(E.earnings_release_dates({"filings": {"recent": {"form": ["10-K"], "items": [""], "filingDate": ["2024-02-15"]}}})) == 0
    known = E.latest_on_or_before(dates, pd.DatetimeIndex(["2024-01-28", "2024-01-29", "2024-06-01"]))
    assert pd.isna(known.iloc[0]) and known.iloc[1] == pd.Timestamp("2024-01-29") and known.iloc[2] == pd.Timestamp("2024-04-25")


class FakeClient:
    """Serves the hand-built fixtures the way EdgarClient would, no network."""

    def __init__(self, facts_by_cik, submissions_by_cik, tickers):
        self.facts_by_cik = facts_by_cik
        self.submissions_by_cik = submissions_by_cik
        self.tickers = tickers
        self.requested = []

    def ticker_map(self):
        return dict(self.tickers)

    def company_facts(self, cik, symbol=None):
        self.requested.append(("facts", cik, symbol))
        return self.facts_by_cik[cik]

    def submissions(self, cik, symbol=None, since=None):
        self.requested.append(("submissions", cik, symbol, since))
        return self.submissions_by_cik[cik]


def _full_company():
    eps = [_fact(r["start"], r["end"], r["val"] / 100, r["filed"], r["form"]) for r in INCOME]
    shares = [_fact(r["start"], r["end"], 100, r["filed"], r["form"]) for r in INCOME if r["start"] != "2023-01-01" or r["end"] == "2023-03-31"]
    cover = [_fact(None, "2024-04-20", 60, "2024-05-01"), _fact(None, "2024-04-20", 40, "2024-05-01")]
    return _facts(
        NetIncomeLoss=INCOME, Revenues=[dict(r, val=r["val"] * 10) for r in INCOME],
        CostOfRevenue=[dict(r, val=r["val"] * 4) for r in INCOME],
        OperatingIncomeLoss=[dict(r, val=r["val"] * 2) for r in INCOME],
        NetCashProvidedByUsedInOperatingActivities=[dict(r, val=r["val"] * 3) for r in INCOME],
        PaymentsToAcquirePropertyPlantAndEquipment=[dict(r, val=r["val"]) for r in INCOME],
        EarningsPerShareDiluted=eps, WeightedAverageNumberOfDilutedSharesOutstanding=shares,
        StockholdersEquity=[dict(r, val=r["val"] // 2) for r in ASSETS], Assets=ASSETS,
        dei_EntityCommonStockSharesOutstanding=cover,
    )


def test_build_fundamentals_panel_through_a_fake_client():
    facts = _full_company()
    # A company that only tags cover-page shares, one row per share class.
    cover_only = _facts(NetIncomeLoss=INCOME, Assets=ASSETS, StockholdersEquity=ASSETS,
                        dei_EntityCommonStockSharesOutstanding=[_fact(None, "2024-04-20", 60, "2024-05-01"),
                                                                _fact(None, "2024-04-20", 40, "2024-05-01")])
    client = FakeClient({1: facts, 2: cover_only}, {1: SUBMISSIONS, 2: SUBMISSIONS}, {"TEST": 1, "TWO": 2})
    dates = pd.bdate_range("2023-01-02", "2024-06-28")
    splits = pd.DataFrame(np.nan, index=dates, columns=["TEST", "TWO", "NOPE"])
    splits.loc[pd.Timestamp("2024-03-01"), "TEST"] = 2.0
    panel = E.build_fundamentals_panel(["TEST", "TWO", "NOPE"], dates, client, splits=splits)
    assert set(panel) == set(E.PANEL_FIELDS)
    for field, frame in panel.items():
        assert list(frame.columns) == ["TEST", "TWO", "NOPE"] and frame.index.equals(dates)
    # The symbol missing from the SEC table is NaN throughout, nothing raised.
    assert panel["net_income_ttm"]["NOPE"].isna().all() and panel["last_filed"]["NOPE"].isna().all()
    # Symbols reach the client with the symbol hint and the panel start as `since`.
    assert ("facts", 1, "TEST") in client.requested and ("submissions", 1, "TEST", dates[0]) in client.requested
    ni = panel["net_income_ttm"]["TEST"]
    assert np.isnan(ni.loc["2024-02-14"]) and ni.loc["2024-02-15"] == 460
    assert panel["revenue_ttm"]["TEST"].loc["2024-02-15"] == 4600
    assert panel["gross_profit_ttm"]["TEST"].loc["2024-02-15"] == 4600 - 1840     # no GrossProfit tag: revenue minus cost
    assert panel["cfo_ttm"]["TEST"].loc["2024-02-15"] == 1380 and panel["capex_ttm"]["TEST"].loc["2024-02-15"] == 460
    assert panel["operating_income_ttm"]["TEST"].loc["2024-02-15"] == 920
    assert panel["equity"]["TEST"].loc["2024-02-15"] == 600 and panel["assets"]["TEST"].loc["2024-02-15"] == 1200
    assert panel["assets_lag4q"]["TEST"].loc["2024-05-01"] == 1010
    # Shares: 100 in every filing; those filed before the 2:1 split of
    # 2024-03-01 are restated to 200 in today's basis, EPS to half.
    assert panel["shares"]["TEST"].loc["2024-02-15"] == 200 and panel["shares"]["TEST"].loc["2024-05-01"] == 100
    assert panel["eps_diluted_q"]["TEST"].loc["2024-02-14"] == pytest.approx(1.2 / 2)
    assert panel["eps_diluted_ttm"]["TEST"].loc["2024-05-01"] == pytest.approx(1.4 + (1.1 + 1.2 + 1.3) / 2)
    assert ni.loc["2024-05-01"] == 140 + 130 + 120 + 110
    assert panel["shares_lag4q"]["TEST"].loc["2024-05-01"] == 100
    # The cover-page fallback sums the share classes.
    assert panel["shares"]["TWO"].loc["2024-05-01"] == 100 and np.isnan(panel["shares"]["TWO"].loc["2024-04-30"])
    assert panel["last_filed"]["TEST"].loc["2024-03-01"] == pd.Timestamp("2024-02-15")
    assert panel["last_earnings_release"]["TEST"].loc["2024-04-26"] == pd.Timestamp("2024-04-25")
    assert pd.isna(panel["last_earnings_release"]["TEST"].loc["2024-01-26"])
    assert panel["sue"]["TEST"].isna().all()   # far too short a history


def test_fundamental_factors_signs_and_market_cap():
    dates = pd.DatetimeIndex(["2024-06-03", "2024-06-04"])
    cols = ["GOOD", "BAD"]

    def frame(good, bad):
        return pd.DataFrame({"GOOD": [good, good], "BAD": [bad, bad]}, index=dates, dtype=float)

    panel = {
        "net_income_ttm": frame(100, 10), "revenue_ttm": frame(1000, 1000), "revenue_ttm_lag4q": frame(800, 1000),
        "gross_profit_ttm": frame(500, 200), "operating_income_ttm": frame(200, 50),
        "cfo_ttm": frame(150, 5), "capex_ttm": frame(20, 20), "eps_diluted_ttm": frame(1, 0.1), "eps_diluted_q": frame(0.3, 0.02),
        "sue": frame(2.0, -1.0), "equity": frame(500, 100), "assets": frame(1000, 1000), "assets_lag4q": frame(1000, 800),
        "shares": frame(100, 100), "shares_lag4q": frame(110, 90),
        "last_filed": pd.DataFrame({"GOOD": pd.to_datetime(["2024-05-01"] * 2), "BAD": pd.to_datetime(["2024-05-01"] * 2)}, index=dates),
        "last_earnings_release": pd.DataFrame({"GOOD": pd.to_datetime(["2024-06-01"] * 2), "BAD": pd.to_datetime(["2024-04-01", pd.NaT])}, index=dates),
    }
    prices = pd.DataFrame({"GOOD": [10.0, 10.0], "BAD": [10.0, 10.0]}, index=dates)
    f = E.fundamental_factors(panel, prices)
    expected = {"earnings_yield", "book_to_market", "cfo_yield", "fcf_yield", "gross_profitability", "roe", "roa",
                "operating_margin", "accruals", "asset_growth", "net_issuance", "sue", "revenue_growth", "days_since_earnings"}
    assert set(f) == expected
    for name, frame_ in f.items():
        assert list(frame_.columns) == cols and frame_.index.equals(dates)
        assert frame_["GOOD"].iloc[0] > frame_["BAD"].iloc[0] or (name == "days_since_earnings" and pd.isna(frame_["BAD"].iloc[1])), name
    assert f["earnings_yield"]["GOOD"].iloc[0] == pytest.approx(100 / (10 * 100))
    assert f["accruals"]["GOOD"].iloc[0] == pytest.approx(-(100 - 150) / 1000)
    assert f["net_issuance"]["GOOD"].iloc[0] == pytest.approx(-(100 / 110 - 1))
    assert f["revenue_growth"]["GOOD"].iloc[0] == pytest.approx(0.25)
    assert f["days_since_earnings"]["GOOD"].tolist() == [-2.0, -3.0]
    assert f["days_since_earnings"]["BAD"].iloc[0] == -63.0 and np.isnan(f["days_since_earnings"]["BAD"].iloc[1])
    # Non-positive denominators are NaN, never a flipped sign.
    panel["equity"] = frame(-5, 100)
    assert np.isnan(E.fundamental_factors(panel, prices)["roe"]["GOOD"].iloc[0])


def test_client_reads_the_cache_and_never_fetches_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="user agent"):
        E.EdgarClient()
    monkeypatch.setenv("EDGAR_USER_AGENT", "tests tests@example.com")
    client = E.EdgarClient(cache_dir=str(tmp_path), offline=True)
    assert client.user_agent == "tests tests@example.com"

    def write(name, payload):
        with gzip.open(tmp_path / name, "wb") as fh:
            fh.write(json.dumps(payload).encode())

    write("TEST_facts.json.gz", _full_company())
    write("CIK0000000002_facts.json.gz", {"cik": 2, "facts": {}})
    write("company_tickers.json.gz", {"0": {"cik_str": 1, "ticker": "TEST", "title": "Test Co"},
                                      "1": {"cik_str": 3, "ticker": "BRK-B", "title": "Berkshire"}})
    assert client.company_facts(1, symbol="TEST")["entityName"] == "Test Co"   # sample-style name through the hint
    assert client.company_facts(2)["cik"] == 2                                   # CIK-style name
    with pytest.raises(FileNotFoundError):
        client.company_facts(4, symbol="MISSING")
    tickers = client.ticker_map()
    assert tickers["TEST"] == 1 and tickers["BRK-B"] == 3 and tickers["BRK.B"] == 3 and tickers["BRKB"] == 3
    assert E.resolve_cik("brk-b", tickers) == 3 and E.resolve_cik("BRK.B", tickers) == 3
    with pytest.raises(KeyError):
        E.resolve_cik("ZZZZ", tickers)

    # The filing index: pages are merged back to `since`, a missing page
    # just ends the merge.
    main = {"cik": "0000000001", "filings": {"recent": {"form": ["10-Q"], "filingDate": ["2024-05-01"], "items": [""]},
                                             "files": [{"name": "CIK0000000001-submissions-001.json", "filingFrom": "2020-01-01", "filingTo": "2024-04-30"},
                                                       {"name": "CIK0000000001-submissions-002.json", "filingFrom": "2010-01-01", "filingTo": "2019-12-31"}]}}
    write("CIK0000000001_submissions.json.gz", main)
    write("CIK0000000001_submissions-001.json.gz", {"form": ["8-K", "8-K"], "filingDate": ["2024-01-29", "2021-02-01"], "items": ["2.02", "2.02"]})
    merged = client.submissions(1, since="2023-01-01")
    assert merged["filings"]["recent"]["form"] == ["10-Q", "8-K", "8-K"]
    assert list(E.earnings_release_dates(merged).strftime("%Y-%m-%d")) == ["2021-02-01", "2024-01-29"]
    deeper = client.submissions(1, since="2015-01-01")                           # page 002 is not cached: stop there
    assert deeper["filings"]["recent"]["filingDate"] == ["2024-05-01", "2024-01-29", "2021-02-01"]
    assert client.submissions(1)["filings"]["recent"]["form"] == ["10-Q"]


def test_merge_company_facts_appends_predecessor_rows():
    new = _facts(NetIncomeLoss=[_fact("2025-01-01", "2025-03-31", 5, "2025-05-01")])
    old = _facts(NetIncomeLoss=[_fact("2024-01-01", "2024-03-31", 4, "2024-05-01")], Assets=[_fact(None, "2024-03-31", 9, "2024-05-01")])
    merged = E.merge_company_facts(new, old)
    rows = E.extract_facts(merged, ["NetIncomeLoss"])
    assert rows["val"].tolist() == [4, 5]
    assert len(E.extract_facts(merged, ["Assets"])) == 1
    assert new["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"][0]["val"] == 5   # inputs untouched


@pytest.mark.skipif(not os.path.exists(f"{SAMPLE_DIR}/AAPL_facts.json.gz"), reason="EDGAR sample not on disk")
def test_real_apple_sample_is_sane():
    with gzip.open(f"{SAMPLE_DIR}/AAPL_facts.json.gz") as fh:
        facts = json.load(fh)
    rows = E.extract_facts(facts, E.CONCEPTS["net_income"])
    quarters = E.discrete_quarters(rows)
    latest = quarters.sort_values(["end", "filed"]).drop_duplicates("end", keep="last")
    ends = latest.loc[(latest["end"] >= "2012-01-01") & (latest["end"] <= "2025-12-31"), "end"]
    # Fourteen years of four quarters, Q4s derived, and no gap in the chain
    # (Apple's fiscal quarters end on Saturdays, so calendar quarters are
    # not the right unit).
    assert len(ends) == 56
    gaps = ends.diff().dt.days.dropna()
    assert gaps.between(75, 125).all()
    assert latest.loc[latest["end"].isin(ends), "derived"].sum() == 14
    events = E.ttm(quarters)
    dates = pd.bdate_range("2009-01-01", "2026-10-21")
    s = E.as_of(events, dates)
    fy2024 = 93.736e9
    assert abs(s.loc["2024-11-04"] / fy2024 - 1) < 0.03
    # The fiscal 2024 figure appears exactly on the 10-K's filing date.
    assert s.loc["2024-10-31"] != pytest.approx(fy2024, rel=0.03)
    assert s[s.round(-6) == round(fy2024, -6)].index[0] == pd.Timestamp("2024-11-01")
    # Nothing is known before it was filed: every as-of value on every date
    # is one of the values filed on or before that date.
    sample = dates[::97]
    for d in sample:
        v = s.loc[d]
        if np.isnan(v):
            continue
        assert v in set(events.loc[events["filed"] <= d, "val"])


def test_research_script_runs_end_to_end_offline(tmp_path, monkeypatch):
    """research/edgar_fundamentals.py with every network call replaced."""
    from research import edgar_fundamentals as script
    from research import factor_lab as lab

    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2019-01-02", periods=700)
    symbols = [f"S{i:02d}" for i in range(25)]
    rets = rng.normal(0.0004, 0.02, size=(len(dates), len(symbols)))
    closes = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=symbols)
    closes["BENCH"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=len(dates)))
    opens = closes[symbols].shift(1).bfill()
    volumes = pd.DataFrame(1e6, index=dates, columns=symbols)
    mask = pd.DataFrame(True, index=dates, columns=symbols)
    sector_map = pd.Series(["A"] * 13 + ["B"] * 12, index=symbols)
    monkeypatch.setattr(lab, "fetch_universe", lambda *a, **k: (closes, opens, volumes, "BENCH", symbols, mask, sector_map))
    monkeypatch.setattr(script, "load_config", lambda path: {})
    monkeypatch.setattr(script, "fetch_splits_and_raw_closes", lambda *a, **k: (None, None))

    class Client:
        def __init__(self, *a, **k):
            pass

        def ticker_map(self):
            return {s: i + 1 for i, s in enumerate(symbols)}

    monkeypatch.setattr(script, "EdgarClient", Client)

    def fake_panel(syms, idx, client, ticker_map=None, splits=None, predecessors=None):
        assert list(syms) == symbols and predecessors == script.PREDECESSOR_CIKS
        panel = {}
        for field in E.PANEL_FIELDS:
            if field in ("last_filed", "last_earnings_release"):
                stamp = pd.DataFrame({s: idx - pd.Timedelta(days=int(d)) for s, d in zip(syms, rng.integers(1, 80, len(syms)))}, index=idx)
                panel[field] = stamp
            else:
                level = rng.lognormal(10, 1, size=len(syms))
                panel[field] = pd.DataFrame(level * (1 + 0.01 * rng.standard_normal((len(idx), len(syms)))), index=idx, columns=syms)
        panel["sue"] = pd.DataFrame(rng.standard_normal((len(idx), len(syms))), index=idx, columns=syms)
        return panel

    monkeypatch.setattr(script, "build_fundamentals_panel", fake_panel)
    (tmp_path / "research" / "out").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["edgar_fundamentals.py", "--universe", "static", "--start", "2019-01-01", "--horizons", "21,10"])
    assert script.main() == 0
    table = pd.read_csv(tmp_path / "research/out/edgar_fundamentals_static.csv", index_col=0)
    assert set(table.index) == {"earnings_yield", "book_to_market", "cfo_yield", "fcf_yield", "gross_profitability", "roe",
                                "roa", "operating_margin", "accruals", "asset_growth", "net_issuance", "sue",
                                "revenue_growth", "days_since_earnings"}
    assert {"raw_ic", "raw_t", "raw_ic_21-26", "neu_ic", "coverage"} <= set(table.columns)
    assert table["coverage"].between(0.99, 1.0).all()
    horizons = pd.read_csv(tmp_path / "research/out/edgar_fundamentals_static_horizons.csv", index_col=0, header=[0, 1])
    assert "h10" in horizons.columns.get_level_values(0)
    coverage = pd.read_csv(tmp_path / "research/out/edgar_coverage_static.csv", index_col=0)
    assert list(coverage.columns) == script.COVERAGE_FIELDS and coverage.iloc[0].between(0.99, 1.0).all()
