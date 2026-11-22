# How we decide / 我们怎么做决策

Capital stance (CEO): **paper only**. Live trading is not enabled.

In the paper phase the “user” is the **operator** (CEO), not the strategy.
We judge whether a book helps the operator decide keep / cut / hold each week —
not whether a long-window backtest looks impressive.

## Line roles / 产品线角色

| Line | Role | Meaning |
|------|------|---------|
| Multi-Factor | **Candidate** | Formal capital candidate on the 100-stock static universe (caveat: static-universe survivorship + leverage explain most backtest excess) |
| Earnings Book | **Candidate (observing)** | Took over the former Trend paper account on 2026-10-23; point-in-time S&P 500 + SUE / YoY EPS growth / 12-6m momentum; accumulating 60-session paper sample |
| LightGBM | **Research** | Paper orders on schedule for telemetry; not competing for Candidate status while validation Rank IC stays near/below zero |
| Trend | **Retired (2026-10-22)** | Disabled (`rebalance-trend.yml` refuses live runs) after trailing the other books; account handed to Earnings Book |

Definitions:

- **Candidate** — weekly decisions may allocate attention and (later) capital narrative here once minimum paper sample and selection attribution hold up.
- **Research** — may run telemetry; excess return claims require selection alpha, not sector luck.
- **Retired** — scheduled execution disabled; retained in code/tests only for historical reproducibility and shared machinery.

## Paper success metrics (USE these)

1. **Gates obeyed** — pre-written stop / upgrade conditions in `docs/STOP_CONDITIONS.md` are checked weekly and not waived casually.
2. **Paper-window alpha** — excess vs SPY on the paper window, with attribution residual visible (not buried in “selection”).
3. **Execution health** — Actions success/fail, stand-down rate (`STAND_DOWN:*`), reconcile drift, stop-loss trigger rate, turnover discipline.
4. **Model & signal health** — `val_rank_ic` / training diagnostics for LGBM; Yahoo earnings calendar coverage (`>= 90%`) and sector concentration for Earnings Book.
5. **Written weekly decision** — keep / cut / hold per active line in `docs/weekly/` (see template).

## Do NOT use for weekly capital narrative

- 5-year Sharpe or long-window Total Return as the headline
- Paper absolute P&L alone
- Backtests without costs, without out-of-sample discipline, or with unresolved survivorship bias
- Comparing Candidate and Research lines as if they were equal product candidates

Dashboard long-window charts (often from 2010/2016) and README honest window slices are **different contracts**. Prefer paper-window + attribution for weekly calls.

## Future live metrics (placeholder only)

When (and only when) live is explicitly approved: net alpha after real costs, slippage vs model, path-dependent drawdown hits, plus compliance / reputation checks.
Do **not** treat paper numbers as live-ready. Paper fills omit real spread/impact.

## Weekly ritual

1. Fill `docs/weekly/TEMPLATE.md` → dated file under `docs/weekly/`.
2. Check stop conditions for active Candidate (observing) and Research books.
3. Record keep / cut / hold for Candidates; Research gets hold or pause recommendations only.
4. Open questions for CEO go in the card — not buried in commit messages.

See also: [STOP_CONDITIONS.md](./STOP_CONDITIONS.md), dashboards https://yingwang.github.io/trade/ · [/earnings/](https://yingwang.github.io/trade/earnings/) · [/lgbm/](https://yingwang.github.io/trade/lgbm/)
