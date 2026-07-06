#!/usr/bin/env python3
"""Re-run backtests ending TODAY and print README-ready markdown tables.

Runs on GitHub Actions via the "README Backtest Refresh" workflow
(readme-backtest.yml, workflow_dispatch) — the tables land in the job
summary and the 5-year chart is uploaded as an artifact. Paste the tables
into README.md; no numbers are invented anywhere, everything is computed
from a live run.

Local usage (only if Actions is unavailable):

    python refresh_backtest_tables.py            # multi-factor only
    python refresh_backtest_tables.py --lgbm     # also LightGBM
"""
import argparse
from datetime import date
from dateutil.relativedelta import relativedelta

from quant.utils.config import load_config
from quant.strategy import MultiFactorStrategy


def _m(metrics, *keys, default=None):
    """Fetch first matching metric key (handles naming drift)."""
    for k in keys:
        if k in metrics:
            return metrics[k]
    return default


def _spy_total_return(result):
    bc = result.benchmark_curve
    if bc is None or len(bc) < 2:
        return None
    return bc.iloc[-1] / bc.iloc[0] - 1.0


def run_window(make_strategy, start, end):
    strat = make_strategy()
    r = strat.run_backtest(start=start, end=end)
    mt = r.metrics
    strat_ret = _m(mt, "Total Return")
    spy_ret = _spy_total_return(r)
    diff = (strat_ret - spy_ret) if (strat_ret is not None and spy_ret is not None) else None
    return {
        "total": strat_ret,
        "spy": spy_ret,
        "diff": diff,
        "cagr": _m(mt, "CAGR"),
        "sharpe": _m(mt, "Sharpe Ratio", "Sharpe"),
        "sortino": _m(mt, "Sortino Ratio", "Sortino"),
        "mdd": _m(mt, "Max Drawdown"),
        "ir": _m(mt, "Information Ratio"),
    }


def pct(x):
    return "—" if x is None else f"{x*100:+.1f}%"


def pp(x):
    return "—" if x is None else f"{x*100:+.1f}pp"


def num(x):
    return "—" if x is None else f"{x:.2f}"


def table(label, span, res):
    return f"""### {label} ({span})

| Metric / 指标             | Strategy / 策略 | SPY    | Difference / 差异 |
| ----------------------- | ------------- | ------ | --------------- |
| **Total Return / 总收益**  | **{pct(res['total'])}** | {pct(res['spy'])} | **{pp(res['diff'])}** |
| **CAGR / 年化收益**         | **{pct(res['cagr'])}** | — | — |
| **Sharpe Ratio**        | **{num(res['sharpe'])}** | — | — |
| **Sortino Ratio**       | **{num(res['sortino'])}** | — | — |
| **Max Drawdown / 最大回撤** | {pct(res['mdd'])} | — | — |
| **Information Ratio**   | **{num(res['ir'])}** | — | — |
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm", action="store_true", help="also run the LightGBM strategy")
    ap.add_argument("--end", default=date.today().isoformat(), help="end date YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    end = args.end
    end_d = date.fromisoformat(end)
    windows = [
        ("5-Year Backtest", 5),
        ("3-Year Backtest", 3),
        ("1-Year Backtest", 1),
    ]
    config = load_config("config.yaml")

    def run_all(make_strategy, title):
        print("\n" + "=" * 70)
        print(f"## {title} — windows ending {end}")
        print("=" * 70)
        for label, yrs in windows:
            start = (end_d - relativedelta(years=yrs)).isoformat()
            try:
                res = run_window(make_strategy, start, end)
                print("\n" + table(label, f"{start} → {end}", res))
            except Exception as e:
                print(f"\n### {label} ({start} → {end})\n!! failed: {e}\n")

    run_all(lambda: MultiFactorStrategy(config), "Multi-Factor Strategy")

    if args.lgbm:
        from quant.signals.lgbm_strategy import LGBMStrategy
        run_all(
            lambda: LGBMStrategy(
                config, train_window=504, val_window=63,
                pred_horizon=21, retrain_every=3, turnover_penalty=0.1,
            ),
            "LightGBM Strategy",
        )

    print("\nDone. Paste the tables above back to Claude to update README.md.")


if __name__ == "__main__":
    main()
