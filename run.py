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


def _plot_backtest_result(result, args, default_filename: str = "backtest_results.png"):
    """Shared plotting logic for all backtest commands."""
    if not args.plot:
        return

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
    outfile = args.plot_output or default_filename
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved to {outfile}")


def cmd_backtest(args):
    """Run a historical backtest with the multi-factor strategy."""
    config = load_config(args.config)
    strategy = MultiFactorStrategy(config)
    result = strategy.run_backtest(start=args.start, end=args.end)

    print(result.summary())
    _plot_backtest_result(result, args, "backtest_results.png")


def cmd_backtest_ml(args):
    """Run a historical backtest with the ML (TFT) strategy."""
    from quant.signals.ml_strategy import MLStrategy

    config = load_config(args.config)
    strategy = MLStrategy(
        config,
        train_window=args.train_window,
        val_window=args.val_window,
        pred_horizon=args.pred_horizon,
        retrain_every=args.retrain_every,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        device=args.device,
    )
    result = strategy.run_backtest(start=args.start, end=args.end)

    print(result.summary())
    _plot_backtest_result(result, args, "backtest_ml_results.png")


def cmd_backtest_ensemble(args):
    """Run a historical backtest with the dual-strategy ensemble."""
    from quant.strategy_ensemble import StrategyEnsemble

    config = load_config(args.config)
    ensemble = StrategyEnsemble(
        config,
        strategy_a_weight=args.weight_a,
        strategy_b_weight=1.0 - args.weight_a,
        consensus_boost=args.consensus_boost,
    )
    result = ensemble.run_backtest(start=args.start, end=args.end)

    print(result.summary())
    _plot_backtest_result(result, args, "backtest_ensemble_results.png")


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

    # Backtest ML
    bt_ml = sub.add_parser("backtest-ml", help="Backtest ML (TFT) strategy")
    bt_ml.add_argument("--start", help="Start date (YYYY-MM-DD)")
    bt_ml.add_argument("--end", help="End date (YYYY-MM-DD)")
    bt_ml.add_argument("--plot", action="store_true", help="Generate performance plots")
    bt_ml.add_argument("--plot-output", help="Plot output filename")
    bt_ml.add_argument("--train-window", type=int, default=252,
                       help="Training window in trading days (default: 252)")
    bt_ml.add_argument("--val-window", type=int, default=63,
                       help="Validation window in trading days (default: 63)")
    bt_ml.add_argument("--pred-horizon", type=int, default=21,
                       help="Prediction horizon in trading days (default: 21)")
    bt_ml.add_argument("--retrain-every", type=int, default=3,
                       help="Retrain every N rebalances (default: 3)")
    bt_ml.add_argument("--hidden-dim", type=int, default=64,
                       help="TFT hidden dimension (default: 64)")
    bt_ml.add_argument("--seq-len", type=int, default=63,
                       help="Input sequence length (default: 63)")
    bt_ml.add_argument("--device", default="auto",
                       help="PyTorch device: cpu, cuda, mps, auto (default: auto)")
    bt_ml.set_defaults(func=cmd_backtest_ml)

    # Backtest Ensemble
    bt_ens = sub.add_parser("backtest-ensemble", help="Backtest dual-strategy ensemble")
    bt_ens.add_argument("--start", help="Start date (YYYY-MM-DD)")
    bt_ens.add_argument("--end", help="End date (YYYY-MM-DD)")
    bt_ens.add_argument("--plot", action="store_true", help="Generate performance plots")
    bt_ens.add_argument("--plot-output", help="Plot output filename")
    bt_ens.add_argument("--weight-a", type=float, default=0.5,
                        help="Capital weight for factor strategy (default: 0.5)")
    bt_ens.add_argument("--consensus-boost", type=float, default=1.3,
                        help="Extra weight for consensus stocks (default: 1.3)")
    bt_ens.set_defaults(func=cmd_backtest_ensemble)

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
