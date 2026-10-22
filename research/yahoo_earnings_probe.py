"""How deep is yfinance's earnings-date history (announcement dates with EPS
estimate, reported EPS and surprise), and how fast can the universe be pulled?"""
import sys
import time

import yfinance as yf

from quant.utils.config import load_config


def main():
    config = load_config("config.yaml")
    symbols = config["universe"]["symbols"]
    t0 = time.time()
    depth = {}
    for i, sym in enumerate(symbols):
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=100)
        except Exception as e:
            print(f"{sym}: FAILED {e}")
            continue
        if ed is None or ed.empty:
            print(f"{sym}: empty")
            continue
        ed = ed.sort_index()
        have = ed.dropna(subset=["Reported EPS"]) if "Reported EPS" in ed.columns else ed
        depth[sym] = (len(ed), len(have), have.index.min().date() if len(have) else None, have.index.max().date() if len(have) else None)
        if i < 3:
            print(f"{sym}: columns={list(ed.columns)}")
            print(have.tail(4).to_string())
        if (i + 1) % 20 == 0:
            print(f"  {i+1} symbols in {time.time()-t0:.0f}s")
    print(f"\n{len(depth)} of {len(symbols)} symbols returned earnings dates in {time.time()-t0:.0f}s")
    import pandas as pd
    table = pd.DataFrame(depth, index=["rows", "with_actual", "first_actual", "last_actual"]).T
    print(table.sort_values("first_actual").to_string())
    print("\nfirst_actual distribution:", table["first_actual"].astype(str).str[:4].value_counts().sort_index().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
