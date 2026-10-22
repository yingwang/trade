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
        "declared-secret-only": os.environ.get("EDGAR_USER_AGENT") or "",
    }
    urls = [
        "https://www.sec.gov/files/company_tickers.json",
        "https://www.sec.gov/include/ticker.txt",
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://data.sec.gov/submissions/CIK0000320193.json",
        "https://efts.sec.gov/LATEST/search-index?q=%22Apple%20Inc%22&forms=10-K",
    ]
    for label, ua in uas.items():
        if not ua:
            print(f"EDGAR [{label}]: no EDGAR_USER_AGENT secret set, skipped")
            continue
        for url in urls:
            status, raw = fetch(url, ua)
            short = url.split("/", 3)[2] + "/" + url.split("/", 3)[3][:40]
            if status == 200:
                print(f"EDGAR [{label}] {short}: 200 {len(raw)}B")
            else:
                print(f"EDGAR [{label}] {short}: {status} :: {text(raw)}")
            time.sleep(0.5)


def universe_sources():
    """Free point-in-time S&P 500 membership and a broad ticker list."""
    ua = "Mozilla/5.0 (compatible; yingwang-trade research; +https://github.com/yingwang/trade)"
    for url in [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(12-10-2024).csv",
        "https://raw.githubusercontent.com/fja05680/sp500/master/README.md",
    ]:
        status, raw = fetch(url, ua)
        print(f"source {url[:90]}: {status} {len(raw)}B")
        if status == 200 and url.endswith(".csv"):
            lines = raw.decode("utf-8", "ignore").splitlines()
            print("   header:", lines[0][:120]); print("   first:", lines[1][:160]); print("   last:", lines[-1][:160]); print("   rows:", len(lines))
        if status == 200 and url.endswith("README.md"):
            print("   ", raw.decode("utf-8", "ignore")[:600].replace("\n", " | "))
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", storage_options={"User-Agent": ua})
        print(f"wikipedia tables: {len(tables)}; constituents {tables[0].shape} cols={list(tables[0].columns)[:6]}; changes {tables[1].shape} cols={list(tables[1].columns)[:6]}")
        print(tables[1].head(3).to_string())
    except Exception as e:
        print("wikipedia read_html FAILED", e)


def yahoo():
    import yfinance as yf

    t0 = time.time()
    wide = yf.download(["SPY"] + [f"{c}" for c in ("A", "AAL", "AAP", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB", "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY", "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", "GEHC", "GEN", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS")], start="2008-01-01", auto_adjust=True, progress=False, threads=True)
    closes = wide["Close"]
    print(f"yfinance bulk download of {closes.shape[1]} S&P names since 2008: {closes.shape} in {time.time()-t0:.0f}s; "
          f"names with data in 2008: {int(closes.loc['2008'].notna().any().sum())}, in 2016: {int(closes.loc['2016'].notna().any().sum())}, "
          f"all-NaN columns: {int(closes.isna().all().sum())}")

    for sym in ["AAPL", "JPM", "ON", "SNOW", "DUK"]:
        t = yf.Ticker(sym)
        try:
            ed = t.get_earnings_dates(limit=100)
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
    print("== universe sources")
    universe_sources()
    print("== yfinance")
    yahoo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
