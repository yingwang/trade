"""Can this runner read SEC EDGAR, and what do the facts look like?

Fetches the ticker/CIK table, one company's XBRL facts and its filing index,
prints the shapes the point-in-time fundamentals loader will have to parse,
and times a small batch so the full universe pull can be sized.
"""
import json
import os
import sys
import time
import urllib.request

UA = os.environ.get("EDGAR_USER_AGENT") or "yingwang/trade research yingwang@users.noreply.github.com"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip
            raw = gzip.decompress(raw)
        return resp.status, raw, time.time() - t0


def main():
    print("UA:", UA)
    status, raw, dt = get("https://www.sec.gov/files/company_tickers.json")
    tickers = json.loads(raw)
    cik = {v["ticker"]: int(v["cik_str"]) for v in tickers.values()}
    print(f"company_tickers: {status} {len(raw)}B {dt:.1f}s n={len(cik)} AAPL={cik.get('AAPL')} ON={cik.get('ON')} GOOGL={cik.get('GOOGL')}")

    status, raw, dt = get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json")
    facts = json.loads(raw)
    print(f"companyfacts AAPL: {status} {len(raw)}B {dt:.1f}s entity={facts['entityName']} taxonomies={list(facts['facts'])}")
    us = facts["facts"]["us-gaap"]
    dei = facts["facts"].get("dei", {})
    print("us-gaap concepts:", len(us), "| dei:", list(dei))
    wanted = [
        "NetIncomeLoss", "EarningsPerShareDiluted", "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax", "GrossProfit",
        "OperatingIncomeLoss", "StockholdersEquity", "Assets",
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding",
    ]
    for c in wanted:
        if c not in us:
            print(f"  {c}: MISSING")
            continue
        units = list(us[c]["units"])
        rows = us[c]["units"][units[0]]
        print(f"  {c}: units={units} n={len(rows)} first_end={rows[0]['end']} last_end={rows[-1]['end']}")
        for r in rows[-4:]:
            print("     ", {k: r.get(k) for k in ("start", "end", "val", "fy", "fp", "form", "filed", "frame")})
    if "EntityCommonStockSharesOutstanding" in dei:
        rows = dei["EntityCommonStockSharesOutstanding"]["units"]["shares"]
        print("  dei:EntityCommonStockSharesOutstanding n=", len(rows), [{k: r.get(k) for k in ("end", "val", "form", "filed", "frame")} for r in rows[-2:]])

    status, raw, dt = get("https://data.sec.gov/submissions/CIK0000320193.json")
    sub = json.loads(raw)
    recent = sub["filings"]["recent"]
    print(f"submissions AAPL: {status} {len(raw)}B {dt:.1f}s keys={list(recent)}")
    eight = [(d, i) for d, f, i in zip(recent["filingDate"], recent["form"], recent["items"]) if f == "8-K" and "2.02" in (i or "")]
    print("  8-K item 2.02 (earnings releases), latest:", eight[:6], "total", len(eight))
    tenq = [(d, f) for d, f in zip(recent["filingDate"], recent["form"]) if f in ("10-Q", "10-K")]
    print("  10-Q/10-K latest:", tenq[:6], "total", len(tenq))

    # Size the universe pull: ten more companies, respecting 10 requests/s.
    sample = ["MSFT", "NVDA", "JPM", "XOM", "GOOGL", "ON", "UNH", "SNOW", "PLD", "DUK"]
    t0 = time.time()
    total = 0
    for sym in sample:
        c = cik.get(sym)
        if c is None:
            print("  no CIK for", sym)
            continue
        status, raw, dt = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json")
        total += len(raw)
        d = json.loads(raw)
        ni = d["facts"].get("us-gaap", {}).get("NetIncomeLoss", {}).get("units", {}).get("USD", [])
        print(f"  {sym}: {status} {len(raw)/1e6:.1f}MB {dt:.1f}s NetIncomeLoss rows={len(ni)} first_end={ni[0]['end'] if ni else None}")
        time.sleep(0.12)
    print(f"ten companies: {total/1e6:.1f}MB in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
