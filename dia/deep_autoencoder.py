"""
dia/deep_autoencoder.py
───────────────────────
Deep Tabular Neural Autoencoder Engine (PyTorch).
Learns low-dimensional non-linear manifold embeddings, computes per-sample
reconstruction errors, and identifies feature-level anomaly attributions.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger("dia.autoencoder")


class TabularAutoencoderNet(nn.Module):
    """Deep Symmetric Feed-Forward Autoencoder with bottleneck latent layer."""

    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        hidden_dim1 = max(16, min(64, input_dim * 2))
        hidden_dim2 = max(8, min(32, input_dim))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def train_tabular_autoencoder(
    X_processed: np.ndarray,
    feature_names: list[str],
    epochs: int = 30,
    batch_size: int = 32,
    latent_dim: int = 4,
    learning_rate: float = 0.005,
    eval_batch_size: int = 512,
) -> dict[str, Any]:
    """
    Trains a Deep Tabular Autoencoder on normalized feature tensors
    and computes reconstruction error metrics and anomaly flags.
    """
    if len(X_processed) < 10:
        return {
            "status": "insufficient_data",
            "message": "At least 10 samples required to train Deep Autoencoder.",
        }

    # Clamp evaluation batch size to prevent OOM
    eval_batch_size = min(512, max(1, int(eval_batch_size)))

    # NaN and Inf Imputation Guard: cleanly impute NaNs and Infs before tensor conversion
    X_clean = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0)

    # Input Row Capping: sub-sample to 50,000 rows to bound tensor memory
    if len(X_clean) > 50_000:
        rng = np.random.RandomState(42)
        sample_indices = np.sort(rng.choice(len(X_clean), size=50_000, replace=False))
        X_clean = X_clean[sample_indices]

    input_dim = X_clean.shape[1]
    latent_dim = max(2, min(latent_dim, input_dim - 1 if input_dim > 2 else 2))

    X_tensor = torch.tensor(X_clean, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size))

    model = TabularAutoencoderNet(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    loss_history = []

    for _epoch in range(epochs):
        epoch_losses = []
        for (batch_x,) in loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        loss_history.append(float(np.mean(epoch_losses)))

    # Batched evaluation mode in chunks to bound peak tensor memory
    model.eval()
    eval_dataset = TensorDataset(X_tensor)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    recon_chunks: list[np.ndarray] = []
    latent_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for (batch_x,) in eval_loader:
            rec, lat = model(batch_x)
            recon_chunks.append(rec.detach().cpu().numpy())
            latent_chunks.append(lat.detach().cpu().numpy())

    reconstructed_np = np.concatenate(recon_chunks, axis=0) if recon_chunks else np.empty((0, input_dim))
    latent_np = np.concatenate(latent_chunks, axis=0) if latent_chunks else np.empty((0, latent_dim))

    # Per-sample Mean Squared Reconstruction Error
    per_sample_mse = np.mean((X_clean - reconstructed_np) ** 2, axis=1)

    # Per-feature reconstruction error (for attribution)
    per_feature_mse = np.mean((X_clean - reconstructed_np) ** 2, axis=0)

    # Statistical Anomaly Threshold (95th percentile or Mean + 2 * Std)
    anomaly_threshold = float(np.percentile(per_sample_mse, 95))
    is_anomaly = per_sample_mse > anomaly_threshold

    # Feature attribution ranking (safe alignment against input_dim)
    aligned_feature_names = list(feature_names) if feature_names else []
    if len(aligned_feature_names) < input_dim:
        aligned_feature_names += [f"feature_{i}" for i in range(len(aligned_feature_names), input_dim)]
    elif len(aligned_feature_names) > input_dim:
        aligned_feature_names = aligned_feature_names[:input_dim]

    feat_attribution = [
        {"feature": name, "reconstruction_error": round(float(err), 4)}
        for name, err in zip(aligned_feature_names, per_feature_mse)
    ]
    feat_attribution.sort(key=lambda x: x["reconstruction_error"], reverse=True)

    # Explicit resource reclamation
    try:
        del recon_chunks, latent_chunks, X_tensor, dataset, loader, eval_dataset, eval_loader, model, optimizer
    except Exception:
        pass
    gc.collect()

    return {
        "status": "success",
        "input_dimension": input_dim,
        "latent_bottleneck_dimension": latent_dim,
        "training_epochs": epochs,
        "final_reconstruction_loss": round(float(loss_history[-1]), 5),
        "loss_curve": [round(loss_val, 5) for loss_val in loss_history],
        "anomaly_threshold_mse": round(anomaly_threshold, 5),
        "total_anomalies_detected": int(np.sum(is_anomaly)),
        "anomaly_rate_pct": round(float(np.mean(is_anomaly) * 100), 1),
        "per_sample_reconstruction_error": [round(float(e), 4) for e in per_sample_mse[:100]],
        "feature_attribution_ranking": feat_attribution,
        "latent_embeddings_shape": list(latent_np.shape),
    }
