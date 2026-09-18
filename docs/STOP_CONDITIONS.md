# Stop conditions / 可执行停止条件

Checked on the Weekly Decision Card. Numbers are product gates, not marketing.

## LightGBM — Research (dry-run telemetry)

**Operating mode:** scheduled rebalance runs **dry-run only** (no order submit).
Manual `workflow_dispatch` may still use dry-run / status; do not flip schedule back to live paper without an explicit product unfreeze note in this file.

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

Recommend pausing even dry-run Actions if:

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
