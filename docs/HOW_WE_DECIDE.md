# How we decide / 我们怎么做决策

Capital stance (CEO): **paper only**. Live trading is not enabled.

In the paper phase the “user” is the **operator** (CEO), not the strategy.
We judge whether a book helps the operator decide keep / cut / hold each week —
not whether a long-window backtest looks impressive.

## Line roles / 产品线角色

| Line | Role | Meaning |
|------|------|---------|
| Multi-Factor | **Candidate** | Only formal capital candidate under paper stance |
| LightGBM | **Research** | Paper orders on schedule; not competing for Candidate status |
| Trend | **Sandbox** | Params frozen; observe paper sample; no promotion talk yet |

Definitions:

- **Candidate** — weekly decisions may allocate attention and (later) capital narrative here.
- **Research** — may run telemetry; excess return claims require selection alpha, not sector luck.
- **Sandbox** — new or unproven; frozen config; minimum paper sample before “keep” debates.

## Paper success metrics (USE these)

1. **Gates obeyed** — pre-written stop / upgrade conditions in `docs/STOP_CONDITIONS.md` are checked weekly and not waived casually.
2. **Paper-window alpha** — excess vs SPY on the paper window, with attribution residual visible (not buried in “selection”).
3. **Execution health** — Actions success/fail, reconcile drift, stop-loss trigger rate, turnover discipline.
4. **Model health (LGBM)** — `val_rank_ic` / training diagnostics vs the Research gates.
5. **Written weekly decision** — keep / cut / hold per line in `docs/weekly/` (see template).

## Do NOT use for weekly capital narrative

- 5-year Sharpe or long-window Total Return as the headline
- Paper absolute P&L alone
- Backtests without costs, without out-of-sample discipline, or with unresolved survivorship bias
- Comparing three lines as if they were equal product candidates

Dashboard long-window charts (often from 2016) and README honest window slices are **different contracts**. Prefer paper-window + attribution for weekly calls.

## Future live metrics (placeholder only)

When (and only when) live is explicitly approved: net alpha after real costs, slippage vs model, path-dependent drawdown hits, plus compliance / reputation checks.
Do **not** treat paper numbers as live-ready. Paper fills omit real spread/impact.

## Weekly ritual

1. Fill `docs/weekly/TEMPLATE.md` → dated file under `docs/weekly/`.
2. Check stop conditions for Research / Sandbox.
3. Record keep / cut / hold for Candidate; Research/Sandbox get hold or pause recommendations only.
4. Open questions for CEO go in the card — not buried in commit messages.

See also: [STOP_CONDITIONS.md](./STOP_CONDITIONS.md), dashboard https://yingwang.github.io/trade/
