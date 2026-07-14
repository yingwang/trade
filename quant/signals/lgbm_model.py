"""Leakage-aware LightGBM cross-sectional ranking model.

Each trading date is one ranking query.  LightGBM therefore receives query
group sizes and optimizes LambdaRank/NDCG instead of treating correlated
stock/date observations as independent regression rows.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
    SKLEARN_FALLBACK = False
except (ImportError, OSError):
    lgb = None
    LGBM_AVAILABLE = False
    try:
        from sklearn.ensemble import GradientBoostingRegressor as _GBR  # noqa: F401

        SKLEARN_FALLBACK = True
    except ImportError:
        SKLEARN_FALLBACK = False


def purged_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    date_idx: int,
    train_window: int,
    val_window: int,
    pred_horizon: int,
    auxiliary: np.ndarray | None = None,
):
    """Return a purged, embargoed walk-forward train/validation split.

    A target at ``t`` uses prices through ``t + pred_horizon``.  Training rows
    whose labels overlap validation are purged, while validation rows whose
    labels reach the rebalance date are embargoed.  Supplying an auxiliary
    panel (normally forward returns) returns its matching validation slice as
    a fifth element without changing the legacy four-element API.
    """
    val_start = max(0, date_idx - val_window)
    train_start = max(0, val_start - train_window)
    train_end = max(train_start, val_start - pred_horizon)
    val_end = max(val_start, date_idx - pred_horizon)

    if train_end - train_start < 63:
        logger.warning(
            "Insufficient training data after purge: %d days (need >= 63)",
            train_end - train_start,
        )
        return None

    result = (
        X[train_start:train_end],
        y[train_start:train_end],
        X[val_start:val_end],
        y[val_start:val_end],
    )
    if auxiliary is not None:
        result += (auxiliary[val_start:val_end],)
    return result


class LGBMRankingModel:
    """Grouped LambdaRank model with cross-sectional validation diagnostics."""

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        early_stopping_rounds: int = 20,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        label_horizon: int = 21,
        relevance_levels: int = 10,
    ):
        self.params = {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_stopping_rounds,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
        }
        self.label_horizon = max(1, int(label_horizon))
        self.relevance_levels = max(2, min(int(relevance_levels), 31))
        self.model = None
        self.feature_importance_: Optional[pd.Series] = None
        self._train_history: list[dict] = []

    @staticmethod
    def _prepare_panel_data(
        X: np.ndarray,
        y: np.ndarray,
        *,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[int, int]]]:
        """Flatten a panel while retaining one query group per date."""
        rows_x: list[np.ndarray] = []
        rows_y: list[np.ndarray] = []
        groups: list[int] = []
        keys: list[tuple[int, int]] = []

        for ti in range(0, X.shape[0], max(1, stride)):
            valid = np.isfinite(y[ti])
            count = int(valid.sum())
            if count < 2:
                continue
            rows_x.append(X[ti, valid])
            rows_y.append(y[ti, valid].astype(float))
            groups.append(count)
            keys.extend((ti, int(si)) for si in np.flatnonzero(valid))

        if not rows_x:
            feature_count = X.shape[2] if X.ndim == 3 else 0
            return (
                np.empty((0, feature_count)),
                np.empty(0),
                [],
                [],
            )
        return np.vstack(rows_x), np.concatenate(rows_y), groups, keys

    def _relevance_labels(self, y: np.ndarray) -> np.ndarray:
        clipped = np.clip(y, 0.0, 1.0)
        return np.minimum(
            (clipped * self.relevance_levels).astype(int),
            self.relevance_levels - 1,
        )

    def _predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """Predict without sklearn's synthetic feature-name warning."""
        booster = getattr(self.model, "booster_", None)
        if booster is not None:
            best_iteration = getattr(self.model, "best_iteration_", None)
            return np.asarray(
                booster.predict(X, num_iteration=best_iteration or -1), dtype=float
            )
        return np.asarray(self.model.predict(X), dtype=float)

    @staticmethod
    def _ranking_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: list[int],
        forward_returns: np.ndarray | None = None,
    ) -> dict:
        ics: list[float] = []
        spreads: list[float] = []
        return_spreads: list[float] = []
        offset = 0
        for size in groups:
            actual = y_true[offset:offset + size]
            predicted = y_pred[offset:offset + size]
            if np.std(actual) > 0 and np.std(predicted) > 0:
                ic = pd.Series(actual).corr(pd.Series(predicted), method="spearman")
                if pd.notna(ic):
                    ics.append(float(ic))

            order = np.argsort(predicted)
            bucket = max(1, size // 10)
            spreads.append(
                float(actual[order[-bucket:]].mean() - actual[order[:bucket]].mean())
            )
            if forward_returns is not None:
                group_returns = forward_returns[offset:offset + size]
                finite = np.isfinite(group_returns)
                if finite.all():
                    return_spreads.append(
                        float(
                            group_returns[order[-bucket:]].mean()
                            - group_returns[order[:bucket]].mean()
                        )
                    )
            offset += size

        rank_ic = float(np.mean(ics)) if ics else np.nan
        ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan
        result = {
            "val_rank_ic": rank_ic,
            "val_icir": (
                float(rank_ic / ic_std) if np.isfinite(ic_std) and ic_std > 0 else np.nan
            ),
            "val_top_minus_bottom_rank": float(np.mean(spreads)) if spreads else np.nan,
        }
        if return_spreads:
            result["val_top_minus_bottom_return"] = float(np.mean(return_spreads))
        return result

    @staticmethod
    def _groupwise_percentiles(values: np.ndarray, groups: list[int]) -> np.ndarray:
        """Calibrate arbitrary ranker margins to comparable 0-1 ranks."""
        calibrated = np.empty(len(values), dtype=float)
        offset = 0
        for size in groups:
            block = pd.Series(values[offset:offset + size])
            calibrated[offset:offset + size] = block.rank(
                method="average", pct=True
            ).to_numpy()
            offset += size
        return calibrated

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list[str]] = None,
        y_val_returns: np.ndarray | None = None,
    ) -> dict:
        """Fit the model and record ranking skill against a constant baseline."""
        if not LGBM_AVAILABLE and not SKLEARN_FALLBACK:
            return {"status": "no_backend_available"}

        # Daily forward labels overlap heavily.  Keep one independent-ish
        # training query per prediction horizon while retaining every
        # validation date for stable IC diagnostics.
        X_tr, y_tr, train_groups, _ = self._prepare_panel_data(
            X_train, y_train, stride=self.label_horizon
        )
        X_va, y_va, val_groups, val_keys = self._prepare_panel_data(
            X_val, y_val, stride=1
        )
        if len(X_tr) < 100 or len(train_groups) < 3:
            return {
                "status": "insufficient_data",
                "n_train": len(X_tr),
                "train_groups": len(train_groups),
            }

        forward_flat = None
        if y_val_returns is not None and val_keys:
            forward_flat = np.asarray(
                [y_val_returns[ti, si] for ti, si in val_keys], dtype=float
            )

        use_validation = len(X_va) >= 20 and len(val_groups) >= 1
        if LGBM_AVAILABLE:
            self.model = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                importance_type="gain",
                num_leaves=self.params["num_leaves"],
                learning_rate=self.params["learning_rate"],
                n_estimators=self.params["n_estimators"],
                min_child_samples=self.params["min_child_samples"],
                subsample=self.params["subsample"],
                colsample_bytree=self.params["colsample_bytree"],
                reg_alpha=self.params["reg_alpha"],
                reg_lambda=self.params["reg_lambda"],
                random_state=self.params["random_state"],
                verbosity=-1,
                n_jobs=-1,
            )
            callbacks = [lgb.log_evaluation(period=0)]
            fit_kwargs = {
                "X": X_tr,
                "y": self._relevance_labels(y_tr),
                "group": train_groups,
                "callbacks": callbacks,
            }
            if use_validation:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=self.params["early_stopping_rounds"],
                        verbose=False,
                    )
                )
                fit_kwargs.update(
                    eval_set=[(X_va, self._relevance_labels(y_va))],
                    eval_group=[val_groups],
                    eval_names=["validation"],
                    eval_at=[1, 3, 5, 10],
                )
            self.model.fit(**fit_kwargs)
        else:
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(
                n_estimators=min(self.params["n_estimators"], 100),
                max_depth=5,
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
                random_state=self.params["random_state"],
            )
            self.model.fit(X_tr, y_tr)

        best_iter = (
            getattr(self.model, "best_iteration_", None)
            or getattr(self.model, "n_estimators_", None)
            or self.params["n_estimators"]
        )
        train_pred = self._groupwise_percentiles(
            self._predict_matrix(X_tr), train_groups
        )
        info = {
            "status": "ok",
            "objective": "lambdarank" if LGBM_AVAILABLE else "regression_fallback",
            "best_iteration": int(best_iter),
            "n_train": len(X_tr),
            "train_groups": len(train_groups),
            "train_rmse": float(np.sqrt(np.mean((y_tr - train_pred) ** 2))),
            "training_stride": self.label_horizon,
        }
        if use_validation:
            val_pred = self._groupwise_percentiles(
                self._predict_matrix(X_va), val_groups
            )
            val_rmse = float(np.sqrt(np.mean((y_va - val_pred) ** 2)))
            baseline_rmse = float(np.sqrt(np.mean((y_va - 0.5) ** 2)))
            info.update(
                n_val=len(X_va),
                val_groups=len(val_groups),
                val_rmse=val_rmse,
                baseline_rmse=baseline_rmse,
                baseline_skill=(
                    1.0 - val_rmse / baseline_rmse if baseline_rmse > 0 else np.nan
                ),
            )
            info.update(
                self._ranking_metrics(
                    y_va, val_pred, val_groups, forward_returns=forward_flat
                )
            )

        importance = getattr(self.model, "feature_importances_", np.zeros(X_tr.shape[1]))
        names = (
            feature_names[:len(importance)]
            if feature_names is not None and len(feature_names) >= len(importance)
            else list(range(len(importance)))
        )
        self.feature_importance_ = pd.Series(importance, index=names).sort_values(
            ascending=False
        )
        self._train_history.append(info)
        logger.info(
            "Ranking model trained: %d query groups, best_iter=%d, val IC=%s",
            len(train_groups),
            int(best_iter),
            f"{info.get('val_rank_ic'):.4f}"
            if np.isfinite(info.get("val_rank_ic", np.nan))
            else "n/a",
        )
        return info

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")
        return self._predict_matrix(X[-1])

    def predict_ranking(self, X: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata

        raw_scores = self.predict(X)
        ranks = rankdata(raw_scores, method="average")
        return ranks / len(ranks)

    def get_feature_importance(
        self,
        feature_names: Optional[list[str]] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        if self.feature_importance_ is None:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])
        imp = self.feature_importance_.copy()
        if feature_names is not None and len(feature_names) >= len(imp):
            imp.index = feature_names[:len(imp)]
        total = float(imp.sum())
        return pd.DataFrame(
            {
                "feature": imp.index,
                "importance": imp.values,
                "importance_pct": imp.values / total * 100 if total > 0 else 0.0,
            }
        ).head(top_n).reset_index(drop=True)
