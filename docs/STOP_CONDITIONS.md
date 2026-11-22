# Stop conditions / 可执行停止条件

Checked on the Weekly Decision Card. Numbers are product gates, not marketing.

## LightGBM — Research (paper orders)

**Operating mode:** scheduled rebalance runs place **paper orders** (the account is paper; nothing touches real money). Research status limits what the line may claim, not whether its paper book trades.

**Unfreeze note (2026-09-22, CEO):** scheduled runs were dry-run from 2026-09-18 (#13). A dry-run schedule froze the account at its 2026-09-01 book, with stop-losses no longer executed, while the dashboard and the paper attribution kept scoring that frozen basket as the model. The CEO's call: it is paper anyway, so trade it. Paper history from 2026-09-18 to 2026-09-22 reflects the frozen book, not the model.

### Soft hold (stay Research)

Trigger if **any** remain true for **2 consecutive weekly cards**:

1. Honest README backtest windows still fail to beat SPY on excess return (all of 5y/3y/1y negative vs SPY, as currently documented).
2. Latest training refresh `val_rank_ic` **&lt; 0**.
3. Paper attribution (when paper history is still shown) shows excess dominated by **industry** rather than stock selection for the card window.

Action: keep Research label; do not discuss Candidate promotion.

### Hard stop on Candidate promotion

Do **not** promote LGBM to Candidate until **all** hold for **4 consecutive weeks**:

1. At least one honest backtest window shows **positive** excess vs SPY after costs.
2. Rolling `val_rank_ic` **≥ 0.02** on the last two training refreshes.
3. Attribution stock-selection component **&gt; industry** component over the paper window used on the card.

### Pause telemetry (optional escalate)

Recommend pausing the scheduled LGBM Actions if:

- Training / site update fails **3** scheduled attempts in a row, or
- `val_rank_ic` **&lt; −0.05** on **3** consecutive refreshes.

Record the pause on the weekly card; CTO executes the workflow change if needed.

---

## Trend — Sandbox (params frozen)

**Params freeze:** values in `config_trend.yaml` as of the freeze date in that file header.
No same-window parameter sweeps without an explicit **unfreeze** note (date + reason + who) appended here and in the config header.

### No Candidate discussion until sample gate

Do not open “keep as Candidate” debate until **either**:

- **≥ 60** paper trading days of history, **or**
- **≥ 2** full rebalance cycles (10-session cadence) **and** at least one calendar week with SPY 21-day realized vol **&gt; 20%**

Whichever comes first is enough to *start* the discussion; both preferred.

### Sandbox kill / demote to parked

Recommend parking the Trend paper account (stop scheduled rebalance, keep dashboard read-only) if **any**:

1. Paper max drawdown from peak **≥ 20%** before the sample gate is met, or
2. Daily loss breaker / stop-loss path fires **≥ 3** times in **10** trading days, or
3. Trend rebalance or dashboard Actions fail **3** consecutive scheduled attempts.

### Unfreeze rule

Unfreezing params requires: sample gate met **or** a written CEO exception on the weekly card, plus a new freeze snapshot after the change.

## 2026-10-22: trend book retired, account handed to the earnings book; factor-book re-entry cooldown

- The owner retired the trend book as the lowest actual paper return of the three (100,000 on 2026-09-17, 92,707 on 2026-10-22). `rebalance-trend.yml` is disabled, has no schedule and refuses live runs. From 2026-10-23 the same Alpaca paper account trades the earnings book (`rebalance-earnings.yml`, keys `ALPACA_TREND_*` or legacy `ALPACA_CLAUDE_*`); its state file starts empty and does not import the trend book's equity history, and its dashboard starts at 2026-10-23.
- LightGBM stays a Research line. Its workflow was disabled for part of 2026-10-22 while the cooldown below was added, then re-enabled.
- Both factor books now skip names sold on a fixed stop for 5 sessions (`risk.reentry_cooldown_sessions`), backtest and live alike.

---

## Earnings Book — Candidate (observing since 2026-10-23)

**Params freeze:** values in `config_earnings.yaml` as selected on 2026-10-22 (`docs/research/2026-10-22-alpha-audit.md`): point-in-time S&P 500 universe, equal-weighted 3-part composite (`sue`, `eps_growth`, `12-6m momentum`), top 25 equal weights with 40-rank buffer, 5-session rebalance on the anchored calendar, `1.0x` gross exposure, no stops and no regime lights.

### Minimum paper sample before keep / cut debate

Do not judge structural edge or open parameter changes until **≥ 60** paper trading days (~12 five-session rebalance cycles, mid-January 2027), unless a hard stop below fires earlier.

### Hard circuit breaker / park conditions

Recommend parking or halting the Earnings paper book if **any**:

1. **Drawdown breaker (`config_earnings.yaml` `risk.max_drawdown_limit: 0.30`):** paper max drawdown from post-takeover peak (`$97,105.27`) reaches **≥ 30%** (automatically enforced by `check_portfolio_drawdown` on daily runs).
2. **Calendar coverage degradation:** `STAND_DOWN:earnings_coverage` (`< 90%` of priced S&P 500 members returning Yahoo earnings dates) blocks **3** consecutive 5-session rebalance cycles.
3. **Operational failure:** scheduled rebalance or dashboard Actions fail **3** consecutive scheduled days.

### Pre-registered research watch (no live change without unfreeze note)

- `max_sector_weight: 1.0` and `industry_neutral: false` allowed ~44% concentration in Semis / Optical / Storage hardware during Oct–Nov 2026 (21-day drawdown `-20.4%`). Any sector cap or sector-neutral z-score adjustment must first beat or match the 2010–2026 unlevered baseline (`16.7%` CAGR, `0.63` Sharpe, `-34%` MDD) on `research.yml` before an explicit unfreeze note is recorded here.
