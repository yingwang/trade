"""Save a sample of raw EDGAR responses so the point-in-time loader can be
developed and unit-tested offline (the SEC blocks this machine's network,
Actions is allowed through with the declared contact user agent).

Writes research/out/edgar/<name>.json.gz for the ticker table and, for each
sample symbol, the XBRL company facts and the filing index.
"""
import gzip
import json
import os
import sys
import time
import urllib.request

UA = os.environ.get("EDGAR_USER_AGENT")
SAMPLE = ["AAPL", "MSFT", "JPM", "XOM", "GOOGL", "ON", "SNOW", "DUK", "NVDA", "COST", "BRK-B", "META"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw


def main():
    if not UA:
        print("EDGAR_USER_AGENT is not set", file=sys.stderr)
        return 2
    out = "research/out/edgar"
    os.makedirs(out, exist_ok=True)
    raw = get("https://www.sec.gov/files/company_tickers.json")
    with gzip.open(f"{out}/company_tickers.json.gz", "wb") as fh:
        fh.write(raw)
    table = json.loads(raw)
    cik = {v["ticker"].upper(): int(v["cik_str"]) for v in table.values()}
    print("tickers", len(cik))
    for sym in SAMPLE:
        c = cik.get(sym.replace("-", ".")) or cik.get(sym.replace("-", "")) or cik.get(sym)
        if c is None:
            print("no CIK for", sym)
            continue
        for kind, url in (
            ("facts", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json"),
            ("submissions", f"https://data.sec.gov/submissions/CIK{c:010d}.json"),
        ):
            raw = get(url)
            with gzip.open(f"{out}/{sym}_{kind}.json.gz", "wb") as fh:
                fh.write(raw)
            print(f"{sym} CIK{c:010d} {kind}: {len(raw)/1e6:.2f}MB")
            time.sleep(0.15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
