"""Tests for the purged/embargoed walk-forward split (look-ahead guard).

The target at row t is the forward return over [t+1, t+pred_horizon], so a
naive split leaks: validation rows near the rebalance date score against
prices after that date, and the last train rows overlap the val window.
"""

import numpy as np

from quant.signals.lgbm_model import purged_train_val_split


def _panel(T=700, N=4, F=3):
    """Panel where every cell equals its own time index, so window
    boundaries can be asserted exactly."""
    X = np.arange(T, dtype=float)[:, None, None] * np.ones((T, N, F))
    y = np.arange(T, dtype=float)[:, None] * np.ones((T, N))
    return X, y


class TestPurgedSplit:
    def test_val_window_is_embargoed(self):
        X, y = _panel()
        date_idx, train_w, val_w, horizon = 650, 504, 63, 21
        X_tr, y_tr, X_va, y_va = purged_train_val_split(
            X, y, date_idx, train_w, val_w, horizon
        )
        # val = [date_idx - val_w, date_idx - horizon): the last horizon rows
        # before the rebalance date are excluded (their targets use future prices)
        assert X_va[0, 0, 0] == date_idx - val_w          # 587
        assert X_va[-1, 0, 0] == date_idx - horizon - 1   # 628

    def test_train_window_is_purged(self):
        X, y = _panel()
        date_idx, train_w, val_w, horizon = 650, 504, 63, 21
        X_tr, y_tr, X_va, y_va = purged_train_val_split(
            X, y, date_idx, train_w, val_w, horizon
        )
        val_start = date_idx - val_w
        # train ends horizon rows before the val window starts
        assert X_tr[-1, 0, 0] == val_start - horizon - 1  # 565
        # so no train row's target window [t+1, t+horizon] reaches into val
        last_train_t = int(y_tr[-1, 0])
        assert last_train_t + horizon < val_start

    def test_insufficient_data_returns_none(self):
        X, y = _panel(T=100)
        # date_idx 90: purged train window is only 6 rows -> refuse
        assert purged_train_val_split(X, y, 90, 504, 63, 21) is None
