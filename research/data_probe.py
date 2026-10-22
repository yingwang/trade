"""Which free fundamental sources can this runner reach, and how deep do they go?

Tries SEC EDGAR with several User-Agent strings (printing the body of a 403 so
an IP block can be told from a header rule), then yfinance's earnings-date
history (announcement dates, EPS estimate/actual/surprise) and quarterly
statements, reporting rows and date ranges for a handful of names.
"""
import gzip
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import pandas as pd


def fetch(url, ua):
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate", "Host": url.split("/")[2]})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass
        return e.code, raw


def text(raw):
    t = re.sub(r"<style.*?</style>|<script.*?</script>", " ", raw.decode("utf-8", "ignore"), flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", html.unescape(t))[:300]


def edgar():
    uas = {
        "declared": os.environ.get("EDGAR_USER_AGENT") or "yingwang/trade research yingwang@users.noreply.github.com",
        "declared2": "Ying Wang yingwang@users.noreply.github.com",
        "browser": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
        "bot-url": "Mozilla/5.0 (compatible; yingwang-trade/1.0; +https://github.com/yingwang/trade)",
    }
    urls = [
        "https://www.sec.gov/files/company_tickers.json",
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://data.sec.gov/submissions/CIK0000320193.json",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-Q&output=atom",
    ]
    for label, ua in uas.items():
        for url in urls:
            status, raw = fetch(url, ua)
            short = url.split("/", 3)[2] + "/" + url.split("/", 3)[3][:40]
            if status == 200:
                print(f"EDGAR [{label}] {short}: 200 {len(raw)}B")
            else:
                print(f"EDGAR [{label}] {short}: {status} :: {text(raw)}")
            time.sleep(0.5)


def yahoo():
    import yfinance as yf

    for sym in ["AAPL", "JPM", "ON", "SNOW", "DUK"]:
        t = yf.Ticker(sym)
        try:
            ed = t.get_earnings_dates(limit=120)
            if ed is None or ed.empty:
                print(f"yfinance earnings_dates {sym}: empty")
            else:
                ed = ed.sort_index()
                have = ed.dropna(subset=["Reported EPS"]) if "Reported EPS" in ed.columns else ed
                print(f"yfinance earnings_dates {sym}: rows={len(ed)} with_actual={len(have)} "
                      f"{ed.index.min().date()} -> {ed.index.max().date()} cols={list(ed.columns)}")
                print(have.tail(3).to_string())
        except Exception as e:
            print(f"yfinance earnings_dates {sym}: FAILED {e}")
        try:
            q = t.get_income_stmt(freq="quarterly")
            print(f"yfinance quarterly income {sym}: shape={None if q is None else q.shape} cols={[] if q is None else [str(c.date()) for c in q.columns]}")
            b = t.get_balance_sheet(freq="quarterly")
            print(f"yfinance quarterly balance {sym}: shape={None if b is None else b.shape}")
        except Exception as e:
            print(f"yfinance statements {sym}: FAILED {e}")
        time.sleep(1.0)


def main():
    print("== EDGAR")
    edgar()
    print("== yfinance")
    yahoo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
