"""LightGBM ranking model for cross-sectional stock selection.

Uses LightGBM's gradient boosting framework to learn a mapping from the 46
ML features (see ml_features.py) to forward return ranks.  Designed as a
faster, more robust alternative to the TFT model:

  - Trains in seconds (vs. ~40 minutes per window for TFT)
  - Longer training window: 504 days (2 years) vs. 252 days
  - Proper validation set with early stopping to prevent overfitting
  - Built-in feature importance for interpretability
  - No GPU required

Training scheme:
  - Rolling window: 504-day train, 63-day validation, predict 21 days ahead
  - Target: cross-sectional percentile rank of forward 21-day returns
  - Early stopping on validation set (patience=20 rounds)
  - Retrains every 3 rebalances (~63 trading days)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Graceful fallback: try lightgbm first, then sklearn GradientBoosting
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    SKLEARN_FALLBACK = False
    logger.info("Using LightGBM backend")
except (ImportError, OSError):
    lgb = None
    LGBM_AVAILABLE = False
    try:
        from sklearn.ensemble import GradientBoostingRegressor as _GBR
        SKLEARN_FALLBACK = True
        logger.info("lightgbm not available; using sklearn GradientBoostingRegressor fallback")
    except ImportError:
        SKLEARN_FALLBACK = False
        logger.info("Neither lightgbm nor sklearn available; LightGBM strategy will use equal-weight fallback")


class LGBMRankingModel:
    """LightGBM-based cross-sectional stock ranking model.

    Trains a gradient-boosted regression model to predict the cross-sectional
    percentile rank of each stock's forward 21-day return.  At inference time,
    predicted ranks are used directly as alpha scores for portfolio construction.

    Parameters
    ----------
    num_leaves : int
        Maximum number of leaves per tree (controls model complexity).
    learning_rate : float
        Boosting learning rate (shrinkage).
    n_estimators : int
        Maximum number of boosting rounds.
    early_stopping_rounds : int
        Stop training if validation metric does not improve for this many rounds.
    min_child_samples : int
        Minimum number of samples in a leaf (regularization).
    subsample : float
        Fraction of training data used per boosting round.
    colsample_bytree : float
        Fraction of features used per tree.
    reg_alpha : float
        L1 regularization on leaf weights.
    reg_lambda : float
        L2 regularization on leaf weights.
    random_state : int
        Random seed for reproducibility.
    """

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
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_importance_: Optional[pd.Series] = None
        self._train_history: list[dict] = []

    def _prepare_panel_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flatten 3D panel data (T, N, F) + 2D targets (T, N) into 2D tabular form.

        Each (date, stock) pair becomes one row.  Rows with NaN targets
        (future dates where forward returns are unknown) are dropped.

        Parameters
        ----------
        X : ndarray of shape (T, N, F)
            Feature tensor.
        y : ndarray of shape (T, N)
            Target matrix (cross-sectional percentile ranks).

        Returns
        -------
        X_flat : ndarray of shape (samples, F)
        y_flat : ndarray of shape (samples,)
        """
        T, N, F = X.shape
        X_flat = X.reshape(T * N, F)
        y_flat = y.reshape(T * N)

        # Drop rows where target is exactly 0 and features are all 0
        # (these are padding / missing data rows)
        valid_mask = np.isfinite(y_flat) & (y_flat != 0.0)
        # Also keep rows where target is 0 but features are non-zero
        # (a stock can legitimately have the lowest rank)
        has_features = np.any(X_flat != 0, axis=1)
        valid_mask = valid_mask | (np.isfinite(y_flat) & has_features)

        return X_flat[valid_mask], y_flat[valid_mask]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """Train the LightGBM model on a rolling window.

        Parameters
        ----------
        X_train : ndarray of shape (T_train, N, F)
            Training features.
        y_train : ndarray of shape (T_train, N)
            Training targets (cross-sectional percentile ranks).
        X_val : ndarray of shape (T_val, N, F)
            Validation features (last 63 days of the rolling window).
        y_val : ndarray of shape (T_val, N)
            Validation targets.
        feature_names : list of str, optional
            Feature names for importance tracking.

        Returns
        -------
        dict with training info: status, best_iteration, train_rmse, val_rmse.
        """
        if not LGBM_AVAILABLE and not SKLEARN_FALLBACK:
            return {"status": "no_backend_available"}

        # Flatten panel data to tabular form
        X_tr, y_tr = self._prepare_panel_data(X_train, y_train)
        X_va, y_va = self._prepare_panel_data(X_val, y_val)

        if len(X_tr) < 100:
            logger.warning(
                "Insufficient training samples: %d (need >= 100). Skipping.",
                len(X_tr),
            )
            return {"status": "insufficient_data", "n_train": len(X_tr)}

        if len(X_va) < 20:
            logger.warning(
                "Insufficient validation samples: %d (need >= 20). "
                "Training without early stopping.",
                len(X_va),
            )
            X_va, y_va = None, None

        backend = "lightgbm" if LGBM_AVAILABLE else "sklearn"
        logger.info(
            "Training %s: %d train samples, %s val samples, %d features",
            backend,
            len(X_tr),
            len(X_va) if X_va is not None else "no",
            X_tr.shape[1],
        )

        if LGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
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
            if X_va is not None:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=self.params["early_stopping_rounds"],
                        verbose=False,
                    )
                )

            fit_kwargs = {
                "X": X_tr,
                "y": y_tr,
                "callbacks": callbacks,
            }
            if feature_names is not None:
                fit_kwargs["feature_name"] = feature_names[:X_tr.shape[1]]

            if X_va is not None:
                fit_kwargs["eval_set"] = [(X_va, y_va)]
                fit_kwargs["eval_names"] = ["validation"]

            self.model.fit(**fit_kwargs)
        else:
            # sklearn GradientBoostingRegressor fallback
            from sklearn.ensemble import GradientBoostingRegressor
            n_est = min(self.params["n_estimators"], 100)  # cap for speed
            self.model = GradientBoostingRegressor(
                n_estimators=n_est,
                max_depth=5,
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
                random_state=self.params["random_state"],
            )
            self.model.fit(X_tr, y_tr)

        # Extract training info
        best_iter = getattr(self.model, "best_iteration_", None) or getattr(self.model, "n_estimators_", None) or self.params["n_estimators"]

        # Compute train and validation RMSE
        train_pred = self.model.predict(X_tr)
        train_rmse = float(np.sqrt(np.mean((y_tr - train_pred) ** 2)))

        info = {
            "status": "ok",
            "best_iteration": best_iter,
            "n_train": len(X_tr),
            "train_rmse": train_rmse,
        }

        if X_va is not None:
            val_pred = self.model.predict(X_va)
            val_rmse = float(np.sqrt(np.mean((y_va - val_pred) ** 2)))
            info["n_val"] = len(X_va)
            info["val_rmse"] = val_rmse

        # Feature importance (gain-based)
        importance = self.model.feature_importances_
        if feature_names is not None and len(feature_names) >= len(importance):
            self.feature_importance_ = pd.Series(
                importance,
                index=feature_names[:len(importance)],
            ).sort_values(ascending=False)
        else:
            self.feature_importance_ = pd.Series(importance).sort_values(ascending=False)

        self._train_history.append(info)
        logger.info(
            "LightGBM trained: best_iter=%d, train_rmse=%.4f%s",
            best_iter,
            train_rmse,
            f", val_rmse={info.get('val_rmse', 0):.4f}" if "val_rmse" in info else "",
        )

        return info

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predicted ranks for the most recent date in the panel.

        Parameters
        ----------
        X : ndarray of shape (T, N, F)
            Feature tensor.  Only the last time step is used for prediction.

        Returns
        -------
        ndarray of shape (N,) with predicted cross-sectional scores.
        Higher values indicate higher expected relative performance.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")

        # Use only the most recent date's features
        X_latest = X[-1]  # shape (N, F)
        predictions = self.model.predict(X_latest)

        return predictions

    def predict_ranking(self, X: np.ndarray) -> np.ndarray:
        """Generate cross-sectional ranking scores (percentile 0-1).

        Parameters
        ----------
        X : ndarray of shape (T, N, F)

        Returns
        -------
        ndarray of shape (N,) with percentile ranks (0=worst, 1=best).
        """
        raw_scores = self.predict(X)
        # Convert to percentile ranks
        from scipy.stats import rankdata
        ranks = rankdata(raw_scores, method="average")
        return ranks / len(ranks)

    def get_feature_importance(
        self,
        feature_names: Optional[list[str]] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Get feature importance from the trained model.

        Parameters
        ----------
        feature_names : list of str, optional
            Names for the features.  If None, uses stored importance.
        top_n : int
            Number of top features to return.

        Returns
        -------
        DataFrame with columns 'feature', 'importance', 'importance_pct'.
        """
        if self.feature_importance_ is None:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

        imp = self.feature_importance_.copy()
        if feature_names is not None and len(feature_names) >= len(imp):
            imp.index = feature_names[:len(imp)]

        total = imp.sum()
        df = pd.DataFrame({
            "feature": imp.index,
            "importance": imp.values,
            "importance_pct": (imp.values / total * 100) if total > 0 else 0,
        })
        return df.head(top_n).reset_index(drop=True)
