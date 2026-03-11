#!/usr/bin/env python3
"""CLI entry point for the quant trading system."""

import argparse
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant.utils.config import load_config
from quant.strategy import MultiFactorStrategy


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_backtest(args):
    """Run a historical backtest."""
    config = load_config(args.config)
    strategy = MultiFactorStrategy(config)
    result = strategy.run_backtest(start=args.start, end=args.end)

    print(result.summary())

    if args.plot:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Equity curve
        ax = axes[0]
        result.equity_curve.plot(ax=ax, label="Strategy", linewidth=1.5)
        if not result.benchmark_curve.empty:
            result.benchmark_curve.plot(ax=ax, label="Benchmark (SPY)", linewidth=1.5, alpha=0.7)
        ax.set_title("Equity Curve")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Drawdown
        ax = axes[1]
        peak = result.equity_curve.cummax()
        dd = (result.equity_curve - peak) / peak
        dd.plot(ax=ax, color="red", linewidth=1)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color="red")
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)

        # Rolling Sharpe
        ax = axes[2]
        rolling_ret = result.returns.rolling(63).mean() * 252
        rolling_vol = result.returns.rolling(63).std() * (252 ** 0.5)
        rolling_sharpe = rolling_ret / rolling_vol
        rolling_sharpe.plot(ax=ax, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Rolling 3-Month Sharpe Ratio")
        ax.set_ylabel("Sharpe")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = args.plot_output or "backtest_results.png"
        plt.savefig(outfile, dpi=150)
        print(f"\nPlot saved to {outfile}")


def cmd_signal(args):
    """Show current alpha signals for the universe."""
    config = load_config(args.config)
    strategy = MultiFactorStrategy(config)

    print("Fetching data and computing signals...")
    signals = strategy.get_current_signal()

    print("\n" + "=" * 50)
    print("CURRENT COMPOSITE ALPHA SIGNALS")
    print("=" * 50)
    for sym, score in signals.items():
        bar = "+" * max(0, int(score * 10)) if score > 0 else "-" * max(0, int(-score * 10))
        print(f"  {sym:6s}  {score:+.4f}  {bar}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Quant Trading System - Medium-Term US Equities"
    )
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="Path to config file")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Backtest
    bt = sub.add_parser("backtest", help="Run historical backtest")
    bt.add_argument("--start", help="Start date (YYYY-MM-DD)")
    bt.add_argument("--end", help="End date (YYYY-MM-DD)")
    bt.add_argument("--plot", action="store_true", help="Generate performance plots")
    bt.add_argument("--plot-output", help="Plot output filename")
    bt.set_defaults(func=cmd_backtest)

    # Signals
    sig = sub.add_parser("signal", help="Show current alpha signals")
    sig.set_defaults(func=cmd_signal)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
