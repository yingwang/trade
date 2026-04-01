"""Simplified Temporal Fusion Transformer (TFT) for cross-sectional stock ranking.

Implements core TFT building blocks in pure PyTorch:
  - Variable Selection Network: learns feature importance
  - Gated Residual Network (GRN): nonlinear feature transformation with gating
  - Multi-head attention over time steps: captures temporal patterns
  - Quantile output head: predicts median + uncertainty (10th, 50th, 90th percentile)

Rolling-window training: train on 252 trading days, predict next 21 days.
Cross-sectional output: ranks stocks by predicted relative performance.

Reference: Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting"
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available; TFT model will fall back to simple ranking")


# ======================================================================
# PyTorch model components
# ======================================================================

if TORCH_AVAILABLE:

    class GatedLinearUnit(nn.Module):
        """GLU activation: splits input in half, applies sigmoid gate to one half."""

        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
            self.gate = nn.Linear(input_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x) * torch.sigmoid(self.gate(x))

    class GatedResidualNetwork(nn.Module):
        """GRN block: the core building block of TFT.

        Architecture: Linear -> ELU -> Linear -> GLU -> Add&Norm (skip connection)
        Optional context vector for conditional processing.
        """

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                     dropout: float = 0.1, context_dim: int = 0):
            super().__init__()
            self.fc1 = nn.Linear(input_dim + context_dim, hidden_dim)
            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.glu = GatedLinearUnit(output_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)

            # Skip connection (project if dimensions differ)
            if input_dim != output_dim:
                self.skip = nn.Linear(input_dim, output_dim)
            else:
                self.skip = None

        def forward(self, x: torch.Tensor,
                    context: Optional[torch.Tensor] = None) -> torch.Tensor:
            residual = x
            if context is not None:
                x = torch.cat([x, context], dim=-1)

            hidden = self.elu(self.fc1(x))
            hidden = self.dropout(self.fc2(hidden))
            hidden = self.glu(hidden)

            if self.skip is not None:
                residual = self.skip(residual)

            return self.layer_norm(hidden + residual)

    class VariableSelectionNetwork(nn.Module):
        """VSN: learns which input features are most important.

        Computes softmax attention weights over features, then applies
        per-feature GRN transformations weighted by importance.
        """

        def __init__(self, num_features: int, hidden_dim: int,
                     dropout: float = 0.1):
            super().__init__()
            self.num_features = num_features
            self.hidden_dim = hidden_dim

            # Flatten all features -> importance weights
            self.importance_grn = GatedResidualNetwork(
                num_features, hidden_dim, num_features, dropout
            )
            # Per-feature transformation GRNs
            self.feature_grns = nn.ModuleList([
                GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
                for _ in range(num_features)
            ])

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Parameters
            ----------
            x : Tensor of shape (..., num_features)

            Returns
            -------
            output : Tensor of shape (..., hidden_dim)
            weights : Tensor of shape (..., num_features) - feature importance
            """
            # Compute importance weights
            weights = torch.softmax(self.importance_grn(x), dim=-1)  # (..., F)

            # Transform each feature independently
            transformed = []
            for i in range(self.num_features):
                feat = x[..., i:i + 1]  # (..., 1)
                transformed.append(self.feature_grns[i](feat))  # (..., hidden_dim)

            # Stack and weight: (..., F, hidden_dim) * (..., F, 1) -> sum over F
            stacked = torch.stack(transformed, dim=-2)  # (..., F, hidden_dim)
            weighted = stacked * weights.unsqueeze(-1)  # broadcast
            output = weighted.sum(dim=-2)  # (..., hidden_dim)

            return output, weights

    class TemporalAttention(nn.Module):
        """Multi-head attention over the time dimension.

        Applies causal (look-back only) attention so the model cannot
        peek at future time steps during training.
        """

        def __init__(self, hidden_dim: int, num_heads: int = 4,
                     dropout: float = 0.1):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, hidden_dim)

            Returns
            -------
            Tensor of shape (batch, seq_len, hidden_dim)
            """
            seq_len = x.size(1)
            # Causal mask: prevent attending to future positions
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_out, _ = self.attention(x, x, x, attn_mask=mask)
            return self.layer_norm(attn_out + x)

    class SimplifiedTFT(nn.Module):
        """Simplified Temporal Fusion Transformer for stock ranking.

        Architecture:
          1. Variable Selection Network (learns feature importance)
          2. Positional encoding (sinusoidal)
          3. GRN encoder layers
          4. Multi-head temporal attention
          5. GRN decoder with quantile output

        Input: (batch, seq_len, num_features) -- one stock's feature history
        Output: (batch, quantiles) -- predicted quantiles of forward return rank
        """

        def __init__(
            self,
            num_features: int,
            hidden_dim: int = 64,
            num_heads: int = 4,
            num_encoder_layers: int = 2,
            dropout: float = 0.1,
            quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        ):
            super().__init__()
            self.num_features = num_features
            self.hidden_dim = hidden_dim
            self.quantiles = quantiles

            # Variable selection
            self.vsn = VariableSelectionNetwork(num_features, hidden_dim, dropout)

            # Positional encoding
            self.pos_encoding = nn.Parameter(
                self._sinusoidal_encoding(512, hidden_dim), requires_grad=False
            )

            # Encoder GRN layers
            self.encoder_grns = nn.ModuleList([
                GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(num_encoder_layers)
            ])

            # Temporal attention
            self.temporal_attention = TemporalAttention(hidden_dim, num_heads, dropout)

            # Decoder: aggregate temporal features -> quantile predictions
            self.decoder_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            self.output_layer = nn.Linear(hidden_dim, len(quantiles))

        @staticmethod
        def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, num_features)

            Returns
            -------
            predictions : Tensor of shape (batch, num_quantiles)
            feature_weights : Tensor of shape (batch, seq_len, num_features)
            """
            batch_size, seq_len, _ = x.shape

            # Variable selection (applied at each time step)
            # Reshape to (batch * seq_len, num_features) for VSN
            x_flat = x.reshape(-1, self.num_features)
            selected, weights = self.vsn(x_flat)
            selected = selected.reshape(batch_size, seq_len, self.hidden_dim)
            weights = weights.reshape(batch_size, seq_len, self.num_features)

            # Add positional encoding
            selected = selected + self.pos_encoding[:seq_len].unsqueeze(0)

            # Encoder GRN layers
            hidden = selected
            for grn in self.encoder_grns:
                hidden = grn(hidden)

            # Temporal attention
            hidden = self.temporal_attention(hidden)

            # Decoder: use the last time step's representation
            last_hidden = hidden[:, -1, :]  # (batch, hidden_dim)
            decoded = self.decoder_grn(last_hidden)

            # Quantile output
            predictions = self.output_layer(decoded)  # (batch, num_quantiles)

            return predictions, weights

    class StockRankingDataset(Dataset):
        """Dataset for per-stock time series windows.

        Each sample is a (seq_len, num_features) window for one stock,
        with a scalar target (cross-sectional rank of forward return).
        """

        def __init__(self, X: np.ndarray, y: np.ndarray,
                     seq_len: int = 63, step: int = 1):
            """
            Parameters
            ----------
            X : ndarray of shape (T, N, F)
                Feature tensor.
            y : ndarray of shape (T, N)
                Target tensor (cross-sectional ranks).
            seq_len : int
                Number of look-back days per sample.
            step : int
                Step size for sliding window.
            """
            self.samples = []
            T, N, F = X.shape

            for t in range(seq_len, T, step):
                if np.any(np.isnan(y[t])):
                    continue
                for n in range(N):
                    x_window = X[t - seq_len:t, n, :]  # (seq_len, F)
                    target = y[t, n]
                    if np.isnan(target) or np.any(np.isnan(x_window)):
                        continue
                    self.samples.append((x_window, target))

            logger.info("Created dataset with %d samples (seq_len=%d, step=%d)",
                        len(self.samples), seq_len, step)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            x, y = self.samples[idx]
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )


# ======================================================================
# Quantile loss
# ======================================================================

def quantile_loss(predictions: "torch.Tensor", targets: "torch.Tensor",
                  quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)) -> "torch.Tensor":
    """Pinball loss for quantile regression.

    Parameters
    ----------
    predictions : Tensor of shape (batch, num_quantiles)
    targets : Tensor of shape (batch,)
    quantiles : tuple of quantile levels
    """
    targets = targets.unsqueeze(1).expand_as(predictions)
    errors = targets - predictions
    losses = []
    for i, q in enumerate(quantiles):
        e = errors[:, i]
        loss = torch.max(q * e, (q - 1) * e)
        losses.append(loss)
    return torch.stack(losses, dim=1).mean()


# ======================================================================
# Model trainer
# ======================================================================

class TFTModelTrainer:
    """Manages TFT model training and inference with rolling windows.

    Parameters
    ----------
    num_features : int
        Number of input features.
    hidden_dim : int
        Hidden dimension for GRN and attention layers.
    seq_len : int
        Look-back window in trading days.
    pred_horizon : int
        Forward prediction horizon in trading days.
    quantiles : tuple
        Quantile levels for output.
    learning_rate : float
        Adam optimizer learning rate.
    batch_size : int
        Training batch size.
    max_epochs : int
        Maximum training epochs per rolling window.
    patience : int
        Early stopping patience (epochs without val improvement).
    device : str
        'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        seq_len: int = 63,
        pred_horizon: int = 21,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 50,
        patience: int = 5,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TFT model training")

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.quantiles = quantiles
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        # Select device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[SimplifiedTFT] = None
        self.feature_importance_: Optional[np.ndarray] = None

        logger.info("TFT trainer initialized: features=%d, hidden=%d, seq_len=%d, "
                     "device=%s", num_features, hidden_dim, seq_len, self.device)

    def _build_model(self) -> "SimplifiedTFT":
        """Create a fresh model instance."""
        model = SimplifiedTFT(
            num_features=self.num_features,
            hidden_dim=self.hidden_dim,
            num_heads=4,
            num_encoder_layers=2,
            dropout=0.1,
            quantiles=self.quantiles,
        )
        return model.to(self.device)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        """Train the model on a rolling window of data.

        Parameters
        ----------
        X_train : ndarray of shape (T_train, N, F)
        y_train : ndarray of shape (T_train, N)
        X_val : ndarray of shape (T_val, N, F), optional
        y_val : ndarray of shape (T_val, N), optional

        Returns
        -------
        dict with training history (losses, best epoch, etc.)
        """
        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                      weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )

        # Build datasets
        train_dataset = StockRankingDataset(X_train, y_train, self.seq_len, step=5)
        if len(train_dataset) == 0:
            logger.warning("Empty training dataset; skipping training")
            return {"status": "empty_dataset"}

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, drop_last=False,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = StockRankingDataset(X_val, y_val, self.seq_len, step=5)
            if len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=False,
                    num_workers=0,
                )

        # Training loop
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.max_epochs):
            # --- Train ---
            self.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds, _ = self.model(batch_x)
                loss = quantile_loss(preds, batch_y, self.quantiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            history["train_loss"].append(avg_train)

            # --- Validate ---
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        preds, _ = self.model(batch_x)
                        loss = quantile_loss(preds, batch_y, self.quantiles)
                        val_losses.append(loss.item())
                avg_val = np.mean(val_losses)
                history["val_loss"].append(avg_val)
                scheduler.step(avg_val)

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                if epoch - best_epoch >= self.patience:
                    logger.info("Early stopping at epoch %d (best=%d, val_loss=%.6f)",
                                epoch, best_epoch, best_val_loss)
                    break
            else:
                # No validation; just track training loss
                scheduler.step(avg_train)
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch

            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f", val_loss={history['val_loss'][-1]:.6f}" if history["val_loss"] else ""
                logger.info("Epoch %d/%d: train_loss=%.6f%s",
                            epoch + 1, self.max_epochs, avg_train, val_str)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        logger.info("Training complete: best_epoch=%d, best_val_loss=%.6f",
                     best_epoch, best_val_loss)

        return {
            "status": "trained",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": history["train_loss"][-1],
            "history": history,
        }

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions for the latest time step.

        Parameters
        ----------
        X : ndarray of shape (T, N, F)
            Feature tensor.  The last `seq_len` rows are used.

        Returns
        -------
        median_preds : ndarray of shape (N,)
            Median (50th percentile) predicted rank scores.
        uncertainty : ndarray of shape (N,)
            Prediction interval width (90th - 10th percentile).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")

        self.model.eval()
        T, N, F = X.shape
        start = max(0, T - self.seq_len)
        x_window = X[start:T]  # (seq_len, N, F)
        actual_len = x_window.shape[0]

        # If we have less than seq_len, pad with zeros at the beginning
        if actual_len < self.seq_len:
            pad = np.zeros((self.seq_len - actual_len, N, F), dtype=np.float32)
            x_window = np.concatenate([pad, x_window], axis=0)

        # Build per-stock input: (N, seq_len, F)
        x_stocks = np.transpose(x_window, (1, 0, 2))  # (N, seq_len, F)
        x_tensor = torch.tensor(x_stocks, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds, weights = self.model(x_tensor)
            preds = preds.cpu().numpy()  # (N, num_quantiles)
            weights = weights.cpu().numpy()  # (N, seq_len, num_features)

        # Store feature importance (average over time steps and stocks)
        self.feature_importance_ = weights.mean(axis=(0, 1))  # (F,)

        # Extract median (index 1 for quantiles=(0.1, 0.5, 0.9))
        q_idx = list(self.quantiles).index(0.5) if 0.5 in self.quantiles else 1
        median_preds = preds[:, q_idx]

        # Uncertainty: 90th - 10th percentile
        if len(self.quantiles) >= 3:
            uncertainty = preds[:, -1] - preds[:, 0]
        else:
            uncertainty = np.zeros(N)

        return median_preds, uncertainty

    def predict_ranking(self, X: np.ndarray) -> np.ndarray:
        """Generate cross-sectional rankings from model predictions.

        Returns
        -------
        ranks : ndarray of shape (N,)
            Percentile ranks (0 to 1) where higher = better predicted performance.
        """
        median_preds, _ = self.predict(X)
        # Convert to percentile ranks
        from scipy.stats import rankdata
        ranks = rankdata(median_preds) / len(median_preds)
        return ranks

    def get_feature_importance(self, feature_names: list[str]) -> pd.Series:
        """Return feature importance scores from the Variable Selection Network.

        Returns
        -------
        Series with feature names as index and importance weights as values.
        """
        if self.feature_importance_ is None:
            return pd.Series(dtype=float)
        importance = pd.Series(
            self.feature_importance_[:len(feature_names)],
            index=feature_names[:len(self.feature_importance_)],
        )
        return importance.sort_values(ascending=False)
