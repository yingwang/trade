"""Point-in-time fundamentals from SEC EDGAR's free XBRL company-facts API.

Why this exists: the quality and value factors are disabled in the live books
because yfinance only offers a current snapshot of the fundamentals, which
leaks the future into a backtest.  EDGAR keeps every number a company ever
filed together with the date it was filed, so a quarter's earnings can be
made known to a backtest only from the day the 10-Q or 10-K hit the SEC.

The shape of the data, as learned from a dozen real company-facts files:

* ``facts[taxonomy][concept]["units"][unit]`` is a list of facts.  A fact with
  ``start`` is a duration (income statement, cash flow); one without is an
  instant (balance sheet).  Every filing re-reports comparative prior periods,
  so the same (start, end) appears several times with different ``filed``
  dates, and later versions can be restatements or split-adjusted re-reports.
  ``fy``/``fp`` describe the filing, not the fact, and are never used here.
* The fourth quarter is usually not reported as a discrete period: the 10-K
  carries the full year, so Q4 is derived as the year minus the quarters
  inside it.  Cash-flow lines are worse: 10-Qs report them year to date only,
  so Q2 and Q3 are derived as differences of cumulative periods as well.
* Companies tag the same line with different concepts (``Revenues``,
  ``SalesRevenueNet``, ``RevenueFromContractWithCustomerExcludingAssessedTax``,
  utilities ``RegulatedAndUnregulatedOperatingRevenue``, banks
  ``RevenuesNetOfInterestExpense``) and switch over the years, so each item is
  an ordered list of candidate concepts merged in priority order.
* Share counts and per-share amounts are stated in the share basis of the
  filing date; a filing after a split re-reports the comparatives in the new
  basis but older periods keep the old one forever.  Given a split history the
  loader rescales those events to today's basis so that they line up with
  split-adjusted price histories.

The as-of rule, used for every panel column: on a date, the value of the
latest period end among the facts filed on or before that date, and among
several filings of that period the latest filed (restatements win once they
are public, never before).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Period lengths in days.  Quarters run from 12 weeks (Costco's first three)
# to 17 weeks (its 53-week-year fourth quarter); the nine-month window starts
# at 240 so that a 36-week retail year-to-date (251/252 days) still counts.
QUARTER = (75, 125)
HALF = (165, 200)
NINE_MONTHS = (240, 290)
YEAR = (350, 380)
# Successive quarter ends of a consecutive chain are this far apart.
CONSECUTIVE = (60, 125)
# A panel value whose period ended longer ago than this is unknown, not
# stale: a 10-K lands within 90 days of the year end and the next 10-Q
# 136 days after it at the latest, so a year-old period means the company
# stopped filing or stopped tagging the concept.
STALE_AFTER_DAYS = 400

_DAY = 86_400_000_000_000  # one day in nanoseconds

# Ordered concept candidates per item.  Earlier entries win when two
# concepts report the same period in the same filing.  The extras beyond
# the usual suspects come from the sample companies: Duke Energy stops
# tagging ``Revenues`` in 2017 and reports ``RegulatedAndUnregulated
# OperatingRevenue`` from 2015, Berkshire and Duke tag part of their revenue
# ``RevenueFromContractWithCustomerIncludingAssessedTax``.
CONCEPTS = {
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenuesNetOfInterestExpense",
        "TotalRevenuesAndOtherIncome",
        "RegulatedAndUnregulatedOperatingRevenue",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets": ["Assets"],
    "shares": ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"],
    # Period-end share counts, the fallbacks when no weighted average is
    # tagged without a class dimension (Alphabet before 2023).
    "shares_outstanding": ["CommonStockSharesOutstanding"],
}
UNITS = {"eps_diluted": ("USD/shares",), "shares": ("shares",), "shares_outstanding": ("shares",)}

PANEL_FIELDS = [
    "net_income_ttm", "revenue_ttm", "revenue_ttm_lag4q", "gross_profit_ttm", "operating_income_ttm",
    "cfo_ttm", "capex_ttm", "eps_diluted_ttm", "eps_diluted_q", "sue", "equity", "assets", "assets_lag4q",
    "shares", "shares_lag4q", "last_filed", "last_earnings_release",
]
PERIODIC_FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT")


class EdgarAccessError(RuntimeError):
    """EDGAR refused the request (403): the user agent is missing or blocked."""


# ----------------------------------------------------------------------
# Client: cache first, network only when a file is missing
# ----------------------------------------------------------------------

class EdgarClient:
    """Reads the ticker table, company facts and filing indexes.

    The SEC asks automated tools to declare themselves in the user agent and
    to stay under ten requests per second; undeclared tools get a 403 page.
    Responses are cached as gzip JSON under ``cache_dir``, named by CIK, and
    the ``<SYMBOL>_facts.json.gz`` names of the offline samples are accepted
    through the ``symbol`` hint.  Nothing touches the network while the cache
    has the file, which is what the tests rely on.
    """

    def __init__(self, user_agent: str | None = None, cache_dir: str = "data/edgar_cache/edgar",
                 min_interval: float = 0.13, max_retries: int = 4, timeout: float = 60.0,
                 offline: bool = False):
        self.user_agent = user_agent or os.environ.get("EDGAR_USER_AGENT")
        if not self.user_agent:
            raise ValueError(
                "EDGAR needs a declared user agent: pass user_agent='name contact@example.com' "
                "or set EDGAR_USER_AGENT (the SEC returns 403 to undeclared tools)"
            )
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        # offline: a file missing from the cache is an error, never a request
        # (tests and development on a machine the SEC blocks).
        self.offline = offline
        self._last_request = 0.0

    # -- cache -----------------------------------------------------------

    def _cache_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, name)

    def _read_cache(self, names) -> dict | None:
        for name in names:
            path = self._cache_path(name)
            if os.path.exists(path):
                with gzip.open(path, "rb") as fh:
                    return json.loads(fh.read())
        return None

    def _write_cache(self, name: str, raw: bytes) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._cache_path(name)
        tmp = path + ".tmp"
        with gzip.open(tmp, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, path)

    def _load(self, url: str, names) -> dict:
        cached = self._read_cache(names)
        if cached is not None:
            return cached
        if self.offline:
            raise FileNotFoundError(f"offline EDGAR client and none of {names} is in {self.cache_dir}")
        raw = self._fetch(url)
        self._write_cache(names[0], raw)
        return json.loads(raw)

    # -- network ---------------------------------------------------------

    def _fetch(self, url: str) -> bytes:
        """One GET with the declared user agent, the rate limit and retries.

        429 and 5xx are retried with doubling waits; a 403 is raised at once
        with the page text because it means the user agent was rejected and
        no retry will change that.
        """
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        )
        attempt = 0
        while True:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    return raw
            except urllib.error.HTTPError as exc:
                body = exc.read()[:500].decode("utf-8", "replace")
                if exc.code == 403:
                    raise EdgarAccessError(f"EDGAR refused {url} (403): {body.strip()}") from None
                if exc.code not in (429,) and exc.code < 500:
                    raise
                reason = f"HTTP {exc.code}"
            except urllib.error.URLError as exc:
                reason = f"network error {exc.reason}"
            attempt += 1
            if attempt > self.max_retries:
                raise RuntimeError(f"EDGAR request failed after {attempt} attempts: {url} ({reason})")
            backoff = 2.0 ** attempt
            logger.warning("EDGAR %s for %s, retrying in %.0fs", reason, url, backoff)
            time.sleep(backoff)

    # -- public ----------------------------------------------------------

    def ticker_map(self) -> dict[str, int]:
        """{TICKER: cik} from the SEC table, plus spelling aliases.

        The SEC writes class shares with a dash (BRK-B, BF-B), the same as
        yfinance, so those work as they are; the dotted and the bare
        spellings used by other sources are added as aliases that never
        overwrite a real ticker.
        """
        table = self._load(TICKERS_URL, ["company_tickers.json.gz"])
        out: dict[str, int] = {}
        for entry in table.values():
            ticker = str(entry["ticker"]).upper().strip()
            if ticker:
                out[ticker] = int(entry["cik_str"])
        for ticker, cik in list(out.items()):
            if "-" in ticker:
                out.setdefault(ticker.replace("-", "."), cik)
                out.setdefault(ticker.replace("-", ""), cik)
        return out

    def company_facts(self, cik: int, symbol: str | None = None) -> dict:
        names = [f"CIK{int(cik):010d}_facts.json.gz"]
        if symbol:
            names.append(f"{symbol}_facts.json.gz")
        return self._load(FACTS_URL.format(cik=int(cik)), names)

    def submissions(self, cik: int, symbol: str | None = None, since=None) -> dict:
        """The filing index; with ``since``, older pages are pulled in too.

        ``filings.recent`` holds the newest thousand or so filings and the
        rest sit in paged files listed under ``filings.files``.  For an
        issuer with heavy prospectus traffic (JPMorgan files thousands of
        424B2s a year) the recent block covers one year, so earnings-release
        dates for a backtest need the pages back to ``since``.  The pages'
        parallel lists are appended to ``filings.recent``.
        """
        cik = int(cik)
        names = [f"CIK{cik:010d}_submissions.json.gz"]
        if symbol:
            names.append(f"{symbol}_submissions.json.gz")
        data = self._load(SUBMISSIONS_URL.format(cik=cik), names)
        if since is None:
            return data
        since = pd.Timestamp(since)
        filings = dict(data.get("filings", {}))
        recent = {k: list(v) for k, v in filings.get("recent", {}).items()}
        pages = sorted(filings.get("files", []) or [], key=lambda p: p.get("filingTo", ""), reverse=True)
        for page in pages:
            oldest = min(recent.get("filingDate", []) or ["9999-12-31"])
            if pd.Timestamp(oldest) <= since:
                break
            name = page["name"]
            suffix = name.rsplit("-", 1)[-1].replace(".json", "")
            page_names = [f"CIK{cik:010d}_submissions-{suffix}.json.gz"]
            if symbol:
                page_names.append(f"{symbol}_submissions-{suffix}.json.gz")
            try:
                extra = self._load(SUBMISSIONS_PAGE_URL.format(name=name), page_names)
            except Exception as exc:  # the recent block is still usable, just shorter
                logger.warning("Filing index page %s unavailable: %s", name, exc)
                break
            extra = extra.get("filings", {}).get("recent", extra) if isinstance(extra, dict) else {}
            for key, values in extra.items():
                if isinstance(values, list):
                    recent.setdefault(key, []).extend(values)
        filings["recent"] = recent
        out = dict(data)
        out["filings"] = filings
        return out


def merge_company_facts(primary: dict, *others: dict) -> dict:
    """Append the facts of predecessor registrants to a company's facts.

    A reorganisation can move a ticker to a new CIK whose history starts at
    the reorganisation (ExxonMobil Holdings in 2025), while the old CIK keeps
    the earlier filings.  The row lists are concatenated per concept and
    unit; ``extract_facts`` later keeps one row per period and filing.
    """
    merged = {"facts": {}}
    merged.update({k: v for k, v in primary.items() if k != "facts"})
    for source in (primary, *others):
        for taxonomy, concepts in source.get("facts", {}).items():
            tax = merged["facts"].setdefault(taxonomy, {})
            for concept, body in concepts.items():
                target = tax.setdefault(concept, {"label": body.get("label"), "units": {}})
                for unit, rows in body.get("units", {}).items():
                    target["units"].setdefault(unit, []).extend(rows)
    return merged


# ----------------------------------------------------------------------
# Pure functions on the facts
# ----------------------------------------------------------------------

FACT_COLUMNS = ["start", "end", "val", "filed", "form", "accn", "concept"]


def _ns(values) -> np.ndarray:
    """Datetimes as datetime64[ns]; pandas 3 keeps whatever resolution it
    parsed, and the int arithmetic below wants one unit."""
    return np.asarray(pd.to_datetime(values)).astype("datetime64[ns]")


def _empty(columns) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c in ("start", "end", "filed") else
                                      "bool" if c == "derived" else "float" if c == "val" else "object")
                         for c in columns})


def extract_facts(facts_json: dict, concepts, units=("USD",), taxonomy: str = "us-gaap") -> pd.DataFrame:
    """Rows of the candidate concepts, merged in priority order.

    Returns start (NaT for instants), end, val, filed, form, accn, concept,
    sorted by filed then end.  When two concepts report the same (start,
    end, filed) the earlier candidate wins; exact duplicates of one concept
    (a 10-Q and an 8-K with the same numbers on the same day) collapse to
    one row.  Facts whose period ends after the filing date are discarded as
    tagging errors, because a single far-future ``end`` would otherwise
    become "the latest period" forever.
    """
    frames = []
    available = facts_json.get("facts", {}).get(taxonomy, {})
    for priority, concept in enumerate(concepts):
        body = available.get(concept)
        if not body:
            continue
        for unit in units:
            rows = body.get("units", {}).get(unit)
            if not rows:
                continue
            frame = pd.DataFrame(rows).reindex(columns=["start", "end", "val", "filed", "form", "accn"])
            frame["concept"] = concept
            frame["priority"] = priority
            frames.append(frame)
    if not frames:
        return _empty(FACT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    for col in ("start", "end", "filed"):
        out[col] = pd.to_datetime(out[col], errors="coerce").astype("datetime64[ns]")
    out["val"] = pd.to_numeric(out["val"], errors="coerce")
    out = out[out["val"].notna() & out["end"].notna() & out["filed"].notna()]
    out = out[out["end"] <= out["filed"] + pd.Timedelta(days=7)]
    out = out.drop_duplicates(["concept", "start", "end", "filed", "val"])
    # Keep a period's rows from the best-ranked concept that reports it in
    # that filing; several rows of one concept with the same key survive
    # (multi-class share counts are summed by the caller).
    best = out.groupby(["start", "end", "filed"], dropna=False)["priority"].transform("min")
    out = out[out["priority"] == best]
    out = out.sort_values(["filed", "end"], kind="stable").reset_index(drop=True)
    return out[FACT_COLUMNS]


def instants(rows: pd.DataFrame) -> pd.DataFrame:
    """Balance-sheet style facts (no start) as events: end, val, filed."""
    if rows.empty:
        return _empty(["end", "val", "filed"])
    inst = rows[rows["start"].isna()]
    return inst[["end", "val", "filed"]].sort_values(["filed", "end"], kind="stable").reset_index(drop=True)


def annual_durations(rows: pd.DataFrame) -> pd.DataFrame:
    """Full-year durations (350 to 380 days): start, end, val, filed."""
    if rows.empty:
        return _empty(["start", "end", "val", "filed"])
    dur = rows[rows["start"].notna()]
    days = (dur["end"] - dur["start"]).dt.days
    year = dur[days.between(*YEAR)]
    return year[["start", "end", "val", "filed"]].sort_values(["filed", "end"], kind="stable").reset_index(drop=True)


def discrete_quarters(rows: pd.DataFrame) -> pd.DataFrame:
    """Quarterly values, reported ones plus those derived from cumulative periods.

    Returns end, start, val, filed, derived.  Reported quarters are the
    durations of a quarter's length.  For every half-year, nine-month and
    annual duration the quarter ending on its end date is derived as the
    cumulative value minus the cumulative value at the previous quarter end,
    where that earlier cumulative is a reported year-to-date figure or, when
    the company only reports discrete quarters, the sum of those quarters
    (so a 10-K's year minus its three quarters gives Q4, and a cash-flow
    statement's nine months minus six months gives Q3).

    Everything is done per filing of the cumulative figure: the components
    are the latest versions filed on or before that filing, so every derived
    version carries the ``filed`` date from which it was knowable, and a
    restated or split-adjusted year yields a new Q4 when the restatement is
    filed.  A re-report of a component alone does not trigger a new version:
    comparatives are re-reported quarter and year-to-date together, and
    mixing a restated quarter with an unrestated cumulative would push the
    restatement into the wrong quarter.  All versions of a period are kept;
    ``as_of`` picks.
    """
    cols = ["end", "start", "val", "filed", "derived"]
    if rows.empty:
        return _empty(cols)
    dur = rows[rows["start"].notna() & rows["end"].notna() & rows["filed"].notna() & rows["val"].notna()]
    if dur.empty:
        return _empty(cols)
    start = _ns(dur["start"]).astype("int64")
    end = _ns(dur["end"]).astype("int64")
    filed = _ns(dur["filed"]).astype("int64")
    val = dur["val"].to_numpy(dtype=float)
    days = (end - start) // _DAY

    # Versions of every duration, keyed by (start, end), sorted by filed.
    versions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i in np.argsort(filed, kind="stable"):
        versions.setdefault((start[i], end[i]), ([], []))
        versions[(start[i], end[i])][0].append(filed[i])
        versions[(start[i], end[i])][1].append(val[i])
    versions = {k: (np.asarray(f, dtype="int64"), np.asarray(v, dtype=float)) for k, (f, v) in versions.items()}
    keys = np.array(list(versions), dtype="int64").reshape(-1, 2)
    key_start, key_end = keys[:, 0], keys[:, 1]
    key_days = (key_end - key_start) // _DAY

    is_quarter = (key_days >= QUARTER[0]) & (key_days <= QUARTER[1])
    quarter_starts_by_end: dict[int, list[int]] = {}
    for s, e in keys[is_quarter]:
        quarter_starts_by_end.setdefault(e, []).append(s)

    def latest(key, at):
        """The value of ``key`` from the latest version filed on or before ``at``."""
        entry = versions.get(key)
        if entry is None:
            return None
        pos = np.searchsorted(entry[0], at, side="right") - 1
        return entry[1][pos] if pos >= 0 else None

    def cumulative(year_start, through, at, depth=0):
        """Value accumulated from ``year_start`` through ``through`` as known at ``at``."""
        if through == year_start - _DAY:
            return 0.0
        reported = latest((year_start, through), at)
        if reported is not None:
            return reported
        if depth >= 4:
            return None
        for q_start in quarter_starts_by_end.get(through, ()):
            if q_start <= year_start:
                continue
            quarter = latest((q_start, through), at)
            if quarter is None:
                continue
            base = cumulative(year_start, q_start - _DAY, at, depth + 1)
            if base is not None:
                return base + quarter
        return None

    is_cumulative = (
        ((key_days >= HALF[0]) & (key_days <= HALF[1]))
        | ((key_days >= NINE_MONTHS[0]) & (key_days <= NINE_MONTHS[1]))
        | ((key_days >= YEAR[0]) & (key_days <= YEAR[1]))
    )
    derived = []
    for s, e in keys[is_cumulative]:
        gap = (e - key_end) // _DAY
        candidates = (key_start >= s) & (key_end < e) & (gap >= QUARTER[0]) & (gap <= QUARTER[1])
        prev_ends = np.unique(key_end[candidates])[::-1]
        if prev_ends.size == 0:
            continue
        for at, total in zip(*versions[(s, e)]):
            for prev in prev_ends:
                base = cumulative(s, prev, at)
                if base is not None:
                    derived.append((e, prev, total - base, at))
                    break

    reported_mask = (days >= QUARTER[0]) & (days <= QUARTER[1])
    reported = pd.DataFrame({
        "end": pd.to_datetime(end[reported_mask]), "start": pd.to_datetime(start[reported_mask]),
        "val": val[reported_mask], "filed": pd.to_datetime(filed[reported_mask]), "derived": False,
    })
    if derived:
        arr = np.array(derived, dtype=object)
        extra = pd.DataFrame({
            "end": pd.to_datetime(arr[:, 0].astype("int64")), "start": pd.to_datetime(arr[:, 1].astype("int64")),
            "val": arr[:, 2].astype(float), "filed": pd.to_datetime(arr[:, 3].astype("int64")), "derived": True,
        })
        # A quarter the company reported discretely in the same filing
        # makes the derived copy redundant (Costco tags its 16-week Q4).
        seen = set(zip(reported["end"], reported["filed"]))
        keep = [(e, f) not in seen for e, f in zip(extra["end"], extra["filed"])]
        out = pd.concat([reported, extra[keep]], ignore_index=True)
    else:
        out = reported
    out = out.sort_values(["end", "filed", "derived"], kind="stable").reset_index(drop=True)
    return out[cols]


def as_of(events: pd.DataFrame, dates, stale_after_days: int | None = None) -> pd.Series:
    """The value known on each date, indexed by ``dates``.

    For a date, among the events filed on or before it, the value of the
    latest ``end``; among several filings of that end, the latest filed.
    Vectorised: sorted by filed, the running maximum of ``end`` gives the
    latest period for every prefix, and the last position where ``end``
    equals that running maximum is the latest filing of it.

    With ``stale_after_days`` a value whose period ended more than that many
    days before the date is NaN instead: a company that stops tagging a
    concept (Berkshire's share count after 2015) or stops filing would
    otherwise keep its last number forever.
    """
    dates = pd.DatetimeIndex(dates)
    out = pd.Series(np.nan, index=dates, dtype=float)
    if events is None or len(events) == 0:
        return out
    ev = events[["end", "val", "filed"]].dropna()
    if ev.empty:
        return out
    ev = ev.sort_values(["filed", "end"], kind="stable")
    filed = _ns(ev["filed"]).astype("int64")
    end = _ns(ev["end"]).astype("int64")
    val = ev["val"].to_numpy(dtype=float)
    running_max = np.maximum.accumulate(end)
    position = np.where(end == running_max, np.arange(len(ev)), -1)
    last_of_max = np.maximum.accumulate(position)
    when = _ns(dates).astype("int64")
    prefix = np.searchsorted(filed, when, side="right") - 1
    known = prefix >= 0
    chosen = last_of_max[prefix[known]]
    values = np.full(len(dates), np.nan)
    values[known] = val[chosen]
    if stale_after_days is not None:
        fresh = np.ones(len(dates), dtype=bool)
        fresh[known] = (when[known] - end[chosen]) <= stale_after_days * _DAY
        values = np.where(fresh, values, np.nan)
    out[:] = values
    return out


def _versions_by_end(events: pd.DataFrame):
    """Sorted unique ends (int ns) and, per end, its versions sorted by filed."""
    ev = events[["end", "val", "filed"]].dropna()
    end = _ns(ev["end"]).astype("int64")
    filed = _ns(ev["filed"]).astype("int64")
    val = ev["val"].to_numpy(dtype=float)
    order = np.lexsort((filed, end))
    end, filed, val = end[order], filed[order], val[order]
    ends, first = np.unique(end, return_index=True)
    bounds = list(first) + [len(end)]
    versions = {ends[i]: (filed[bounds[i]:bounds[i + 1]], val[bounds[i]:bounds[i + 1]]) for i in range(len(ends))}
    return ends, versions


def _values_at(versions, end, at: np.ndarray) -> np.ndarray:
    """The latest version of ``end`` filed on or before each date in ``at`` (NaN when none)."""
    entry = versions.get(end)
    if entry is None:
        return np.full(len(at), np.nan)
    pos = np.searchsorted(entry[0], at, side="right") - 1
    return np.where(pos >= 0, entry[1][np.clip(pos, 0, None)], np.nan)


def _chain_breaks(ends: np.ndarray) -> np.ndarray:
    """Running count of gaps between successive ends that are not one quarter long."""
    gaps = np.diff(ends) // _DAY
    broken = (gaps < CONSECUTIVE[0]) | (gaps > CONSECUTIVE[1])
    return np.concatenate([[0], np.cumsum(broken)])


def _events_frame(rows) -> pd.DataFrame:
    if not rows:
        return _empty(["end", "val", "filed"])
    arr = np.array(rows, dtype=object)
    return pd.DataFrame({
        "end": pd.to_datetime(arr[:, 0].astype("int64")), "val": arr[:, 1].astype(float),
        "filed": pd.to_datetime(arr[:, 2].astype("int64")),
    })


def ttm(quarters: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Trailing sums of ``n`` consecutive quarters as events: end, val, filed.

    For every quarter end whose ``n`` most recent quarters form a consecutive
    chain, one event per filing date at which the sum could change: the
    quarter's own filings and any later re-report of a component, each using
    the latest version of every component filed by then.
    """
    if quarters is None or len(quarters) == 0:
        return _empty(["end", "val", "filed"])
    ends, versions = _versions_by_end(quarters)
    breaks = _chain_breaks(ends)
    rows = []
    for i in range(n - 1, len(ends)):
        if breaks[i] != breaks[i - n + 1]:
            continue
        members = ends[i - n + 1:i + 1]
        own_first = versions[ends[i]][0][0]
        when = np.unique(np.concatenate([versions[e][0] for e in members]))
        when = when[when >= own_first]
        total = np.zeros(len(when))
        for e in members:
            total = total + _values_at(versions, e, when)
        for at, value in zip(when, total):
            if np.isfinite(value):
                rows.append((ends[i], value, at))
    return _events_frame(rows)


def seasonal_sue(quarters: pd.DataFrame, lookback: int = 8, min_obs: int = 6) -> pd.DataFrame:
    """Standardized unexpected earnings of the seasonal random walk.

    For a quarter q: (eps_q - eps_{q-4}) divided by the standard deviation of
    the previous ``lookback`` such differences, of which at least ``min_obs``
    must exist on an unbroken quarterly chain.  Events carry the filing date
    of the quarter's own version or of a re-report of its year-ago quarter.
    """
    if quarters is None or len(quarters) == 0:
        return _empty(["end", "val", "filed"])
    ends, versions = _versions_by_end(quarters)
    breaks = _chain_breaks(ends)
    rows = []
    for i in range(4, len(ends)):
        if breaks[i] != breaks[i - 4]:
            continue
        own_first = versions[ends[i]][0][0]
        when = np.unique(np.concatenate([versions[ends[i]][0], versions[ends[i - 4]][0]]))
        when = when[when >= own_first]
        current = _values_at(versions, ends[i], when) - _values_at(versions, ends[i - 4], when)
        history = []
        for k in range(1, lookback + 1):
            j = i - k
            if j - 4 < 0 or breaks[i] != breaks[j - 4]:
                break
            history.append(_values_at(versions, ends[j], when) - _values_at(versions, ends[j - 4], when))
        if len(history) < min_obs:
            continue
        hist = np.vstack(history)
        count = np.isfinite(hist).sum(axis=0)
        with np.errstate(invalid="ignore"):
            spread = np.nanstd(hist, axis=0, ddof=1)
        ok = (count >= min_obs) & np.isfinite(current) & (spread > 0)
        for at, value, good in zip(when, current / np.where(spread > 0, spread, np.nan), ok):
            if good:
                rows.append((ends[i], value, at))
    return _events_frame(rows)


def lagged(events: pd.DataFrame, quarters_back: int = 4, tolerance_days: int = 45) -> pd.DataFrame:
    """The value ``quarters_back`` quarters earlier, as known at each filing.

    For every end, the end nearest to ``quarters_back`` quarters before it
    (within ``tolerance_days``, which covers 52/53-week years) supplies the
    value; the event is dated by the later of the two periods' filings, so a
    restated or split-adjusted comparative replaces the original once filed.
    Used for asset growth, net share issuance and revenue growth.
    """
    if events is None or len(events) == 0:
        return _empty(["end", "val", "filed"])
    ends, versions = _versions_by_end(events)
    target_gap = int(round(quarters_back * 365.25 / 4)) * _DAY
    tolerance = tolerance_days * _DAY
    rows = []
    for i, e in enumerate(ends):
        target = e - target_gap
        pos = np.searchsorted(ends, target)
        nearest = [c for c in (pos - 1, pos) if 0 <= c < i]
        if not nearest:
            continue
        j = min(nearest, key=lambda c: abs(ends[c] - target))
        if abs(ends[j] - target) > tolerance:
            continue
        own_first = versions[e][0][0]
        when = np.unique(np.concatenate([versions[e][0], versions[ends[j]][0]]))
        when = when[when >= own_first]
        values = _values_at(versions, ends[j], when)
        for at, value in zip(when, values):
            if np.isfinite(value):
                rows.append((e, value, at))
    return _events_frame(rows)


def split_adjusted(events: pd.DataFrame, splits: pd.Series | None, per_share: bool = False) -> pd.DataFrame:
    """Share counts (or per-share amounts) restated to today's share basis.

    A filing states shares in the basis of its filing date, so each event is
    scaled by the product of the split ratios dated after its ``filed``
    (divided, for per-share amounts).  Only then do old share counts line up
    with split-adjusted price histories, and a seasonal EPS difference across
    a split stops looking like a collapse in earnings.
    """
    if events is None or len(events) == 0 or splits is None:
        return events
    ratios = pd.Series(splits).dropna()
    ratios = ratios[(ratios > 0) & (ratios != 1)].sort_index()
    if ratios.empty:
        return events
    split_dates = _ns(ratios.index).astype("int64")
    suffix = np.append(np.cumprod(ratios.to_numpy(dtype=float)[::-1])[::-1], 1.0)
    factor = suffix[np.searchsorted(split_dates, _ns(events["filed"]).astype("int64"), side="right")]
    out = events.copy()
    out["val"] = out["val"].to_numpy(dtype=float) / factor if per_share else out["val"].to_numpy(dtype=float) * factor
    return out


def earnings_release_dates(submissions_json: dict) -> pd.DatetimeIndex:
    """Filing dates of 8-Ks with item 2.02 (results of operations), sorted."""
    recent = submissions_json.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    items = recent.get("items", []) or []
    filed = recent.get("filingDate", []) or []
    dates = [
        d for f, i, d in zip(forms, items, filed)
        if str(f).startswith("8-K") and "2.02" in [part.strip() for part in str(i or "").split(",")]
    ]
    return pd.DatetimeIndex(sorted(set(pd.to_datetime(dates)))).astype("datetime64[ns]")


def latest_on_or_before(known: pd.DatetimeIndex, dates) -> pd.Series:
    """For each date, the latest of ``known`` on or before it (NaT when none)."""
    dates = pd.DatetimeIndex(dates)
    known = pd.DatetimeIndex(known).sort_values()
    if len(known) == 0:
        return pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    pos = np.searchsorted(_ns(known), _ns(dates), side="right") - 1
    values = np.where(pos >= 0, _ns(known)[np.clip(pos, 0, None)], np.datetime64("NaT", "ns"))
    return pd.Series(values.astype("datetime64[ns]"), index=dates)


# ----------------------------------------------------------------------
# One company's columns, then the panel
# ----------------------------------------------------------------------

def symbol_fundamentals(facts: dict, submissions: dict | None, dates, splits: pd.Series | None = None) -> dict:
    """Every panel field for one company as Series over ``dates``."""
    dates = pd.DatetimeIndex(dates)
    items = {name: extract_facts(facts, concepts, UNITS.get(name, ("USD",))) for name, concepts in CONCEPTS.items()}
    quarters = {name: discrete_quarters(items[name]) for name in
                ("net_income", "revenue", "cost_of_revenue", "gross_profit", "operating_income", "cfo", "capex")}

    def known(events):
        return as_of(events, dates, STALE_AFTER_DAYS)

    out: dict[str, pd.Series] = {}
    ttm_events = {name: ttm(q) for name, q in quarters.items()}
    for name in ("net_income", "revenue", "operating_income", "cfo", "capex"):
        out[f"{name}_ttm"] = known(ttm_events[name])
    cost_ttm = known(ttm_events["cost_of_revenue"])
    out["gross_profit_ttm"] = known(ttm_events["gross_profit"]).combine_first(out["revenue_ttm"] - cost_ttm)
    out["revenue_ttm_lag4q"] = known(lagged(ttm_events["revenue"]))

    # Per-share and share-count rows are restated to today's basis before
    # any arithmetic: a year filed after a split minus quarters filed before
    # it would otherwise give a nonsense Q4.
    eps = discrete_quarters(split_adjusted(items["eps_diluted"], splits, per_share=True))
    out["eps_diluted_q"] = known(eps)
    out["eps_diluted_ttm"] = known(ttm(eps))
    out["sue"] = known(seasonal_sue(eps))

    out["equity"] = known(instants(items["equity"]))
    assets = instants(items["assets"])
    out["assets"] = known(assets)
    out["assets_lag4q"] = known(lagged(assets))

    # Shares: the weighted average of the latest quarter or year, in the
    # same filing basis as the EPS; then the cover-page count (one row per
    # class for multi-class issuers, hence the sum) and the balance-sheet
    # count as fallbacks where the weighted average is not tagged.
    weighted = split_adjusted(items["shares"], splits)
    weighted = weighted[weighted["start"].notna()]
    weighted_days = (weighted["end"] - weighted["start"]).dt.days
    weighted = weighted[weighted_days.between(*QUARTER) | weighted_days.between(*YEAR)][["end", "val", "filed"]]
    cover = instants(split_adjusted(
        extract_facts(facts, ["EntityCommonStockSharesOutstanding"], ("shares",), taxonomy="dei"), splits))
    if not cover.empty:
        cover = cover.groupby(["end", "filed"], as_index=False)["val"].sum()
    balance = instants(split_adjusted(items["shares_outstanding"], splits))
    sources = (weighted, cover, balance)
    out["shares"] = _first_available([known(src) for src in sources])
    out["shares_lag4q"] = _first_available([known(lagged(src)) for src in sources])

    periodic = pd.concat([rows[rows["form"].isin(PERIODIC_FORMS)]["filed"] for rows in items.values()])
    out["last_filed"] = latest_on_or_before(pd.DatetimeIndex(periodic.unique()), dates)
    releases = earnings_release_dates(submissions) if submissions else pd.DatetimeIndex([])
    out["last_earnings_release"] = latest_on_or_before(releases, dates)
    return out


def _first_available(series_list) -> pd.Series:
    out = series_list[0]
    for s in series_list[1:]:
        out = out.combine_first(s)
    return out


def resolve_cik(symbol: str, ticker_map: dict) -> int:
    """The CIK for a yfinance-spelled symbol (BRK-B), trying the other spellings."""
    sym = symbol.upper().strip()
    for candidate in (sym, sym.replace("-", "."), sym.replace("-", ""), sym.replace(".", "-")):
        if candidate in ticker_map:
            return int(ticker_map[candidate])
    raise KeyError(f"{symbol} is not in the SEC ticker table")


def build_fundamentals_panel(symbols, dates, client: EdgarClient, ticker_map: dict | None = None,
                             splits: pd.DataFrame | None = None, predecessors: dict | None = None) -> dict:
    """Point-in-time fundamentals, one DataFrame (dates x symbols) per field.

    ``splits`` (dates x symbols of split ratios, as yfinance reports them)
    restates share counts and per-share amounts to today's basis;
    ``predecessors`` maps a symbol to earlier CIKs whose facts are merged
    in.  A symbol whose data cannot be fetched or parsed is logged and left
    NaN rather than aborting the panel.
    """
    dates = pd.DatetimeIndex(dates)
    symbols = list(symbols)
    ticker_map = ticker_map if ticker_map is not None else client.ticker_map()
    predecessors = predecessors or {}
    columns: dict[str, dict[str, pd.Series]] = {field: {} for field in PANEL_FIELDS}
    for symbol in symbols:
        try:
            cik = resolve_cik(symbol, ticker_map)
            facts = client.company_facts(cik, symbol=symbol)
            for earlier in predecessors.get(symbol, []):
                try:
                    facts = merge_company_facts(facts, client.company_facts(earlier))
                except Exception as exc:  # a missing predecessor costs history, not the symbol
                    logger.warning("Predecessor CIK %s of %s unavailable: %s", earlier, symbol, exc)
            try:
                submissions = client.submissions(cik, symbol=symbol, since=dates[0] if len(dates) else None)
            except Exception as exc:
                logger.warning("Filing index of %s unavailable, no earnings-release dates: %s", symbol, exc)
                submissions = None
            split_series = splits[symbol] if splits is not None and symbol in splits.columns else None
            for field, series in symbol_fundamentals(facts, submissions, dates, split_series).items():
                columns[field][symbol] = series
        except Exception as exc:
            logger.warning("EDGAR fundamentals for %s unavailable: %s", symbol, exc)
    panel = {}
    for field in PANEL_FIELDS:
        dtype = "datetime64[ns]" if field in ("last_filed", "last_earnings_release") else float
        frame = pd.DataFrame(columns[field], index=dates) if columns[field] else pd.DataFrame(index=dates)
        panel[field] = frame.reindex(columns=symbols).astype(dtype)
    return panel


def fundamental_factors(panel: dict, prices: pd.DataFrame) -> dict:
    """Factor frames (dates x symbols) from the panel, signed so higher = better.

    Market capitalisation is the same-day price times the as-of share count,
    which is only consistent when the shares were restated to the basis of
    the price history (see ``split_adjusted``).  Ratios with a non-positive
    denominator are left NaN rather than flipping sign.
    """
    shares = panel["shares"]
    px = prices.reindex(index=shares.index, columns=shares.columns)
    market_cap = (px * shares).where(lambda m: m > 0)
    assets = panel["assets"].where(panel["assets"] > 0)
    equity = panel["equity"]
    revenue = panel["revenue_ttm"].where(panel["revenue_ttm"] > 0)
    assets_lag = panel["assets_lag4q"].where(panel["assets_lag4q"] > 0)
    shares_lag = panel["shares_lag4q"].where(panel["shares_lag4q"] > 0)
    revenue_lag = panel["revenue_ttm_lag4q"].where(panel["revenue_ttm_lag4q"] > 0)
    release = panel["last_earnings_release"]
    since_release = (np.asarray(_ns(release.index))[:, None] - release.to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "D")
    return {
        "earnings_yield": panel["net_income_ttm"] / market_cap,
        "book_to_market": equity / market_cap,
        "cfo_yield": panel["cfo_ttm"] / market_cap,
        "fcf_yield": (panel["cfo_ttm"] - panel["capex_ttm"]) / market_cap,
        "gross_profitability": panel["gross_profit_ttm"] / assets,
        "roe": panel["net_income_ttm"] / equity.where(equity > 0),
        "roa": panel["net_income_ttm"] / assets,
        "operating_margin": panel["operating_income_ttm"] / revenue,
        "accruals": -(panel["net_income_ttm"] - panel["cfo_ttm"]) / assets,
        "asset_growth": -(panel["assets"] / assets_lag - 1.0),
        "net_issuance": -(shares / shares_lag - 1.0),
        "sue": panel["sue"],
        "revenue_growth": panel["revenue_ttm"] / revenue_lag - 1.0,
        "days_since_earnings": pd.DataFrame(-since_release, index=release.index, columns=release.columns),
    }
