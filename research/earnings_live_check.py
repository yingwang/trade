"""Exercise the earnings book's live path without a broker: the current
S&P 500 membership, Yahoo's calendar for every member, and the target the
runner would trade today from an empty book.  Prints the book, the signal
parts of its names and how long each step took."""
import logging
import sys
import time

import pandas as pd

from quant.earnings_strategy import EarningsStrategy
from quant.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    config = load_config("config_earnings.yaml")
    strategy = EarningsStrategy(config)
    t0 = time.time()
    portfolio = strategy.get_current_portfolio(capital=100_000.0, prev_weights=None)
    print(f"\ntarget computed in {time.time()-t0:.0f}s; universe {len(strategy.data.symbols)} names; "
          f"prices {strategy.last_prices_.shape} through {strategy.last_prices_.index[-1].date()}")
    print(f"decision: {strategy.last_decision_}")
    pd.set_option("display.width", 200)
    print(portfolio.to_string())
    pre = strategy.precompute(strategy.last_prices_)
    parts = strategy.parts_on(pre, strategy.last_prices_.index[-1]).reindex(portfolio.index)
    print("\nsignal parts of the chosen names:\n")
    print(parts.round(3).to_string())
    print(f"\ngross {portfolio['weight'].sum():.3f}, names {len(portfolio)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
