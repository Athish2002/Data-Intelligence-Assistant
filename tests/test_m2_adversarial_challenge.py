"""
tests/test_m2_adversarial_challenge.py
──────────────────────────────────────
Adversarial Stress Suite for Milestone M2:
1. Autoencoder memory capping at boundaries (50k, 50,001, 75k rows) & sampling determinism.
2. Autoencoder chunked evaluation mathematical equivalence vs unbatched across varying batch sizes.
3. Autoencoder resilience against NaNs, Infs, -Infs, and all-NaN columns.
4. AutoFE candidates_df garbage collection and memory leak checks under repeated invocations.
5. Retrieval model cache eviction idempotency and integration hooks.
"""

import gc
import numpy as np
import pandas as pd
import pytest
import torch

from api.server import trigger_system_garbage_collection as api_gc
from dia.deep_autoencoder import TabularAutoencoderNet, train_tabular_autoencoder
from dia.feature_engineer import auto_engineer_features
from dia.retrieval import _MODEL_CACHE, clear_retrieval_model_cache
from dia.session_manager import BoundedSessionStore, trigger_system_garbage_collection as dia_gc


# ─────────────────────────────────────────────────────────────────────────────
# 1. Autoencoder Memory Capping & Row Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_autoencoder_boundary_exact_50k():
    """Verify input with exactly 50,000 rows is preserved without truncation."""
    rng = np.random.RandomState(42)
    X = rng.randn(50_000, 3).astype(np.float32)
    res = train_tabular_autoencoder(X, ["c1", "c2", "c3"], epochs=1, batch_size=1024, eval_batch_size=1024)
    assert res["status"] == "success"
    assert res["latent_embeddings_shape"][0] == 50_000
    assert np.isfinite(res["final_reconstruction_loss"])


def test_autoencoder_boundary_50001_rows():
    """Verify input with 50,001 rows activates row capping down to exactly 50,000."""
    rng = np.random.RandomState(42)
    X = rng.randn(50_001, 3).astype(np.float32)
    res = train_tabular_autoencoder(X, ["c1", "c2", "c3"], epochs=1, batch_size=1024, eval_batch_size=1024)
    assert res["status"] == "success"
    assert res["latent_embeddings_shape"][0] == 50_000
    assert np.isfinite(res["final_reconstruction_loss"])


def test_autoencoder_sampling_determinism_on_large_inputs():
    """Verify subsampling on >50k rows is deterministic across runs."""
    rng = np.random.RandomState(42)
    X = rng.randn(55_000, 3).astype(np.float32)
    torch.manual_seed(100)
    res1 = train_tabular_autoencoder(X, ["c1", "c2", "c3"], epochs=1, batch_size=1024, eval_batch_size=1024)
    torch.manual_seed(100)
    res2 = train_tabular_autoencoder(X, ["c1", "c2", "c3"], epochs=1, batch_size=1024, eval_batch_size=1024)
    assert res1["latent_embeddings_shape"] == res2["latent_embeddings_shape"]
    assert res1["loss_curve"] == res2["loss_curve"]


def test_autoencoder_insufficient_data_boundary():
    """Verify graceful handling when samples < 10."""
    X = np.ones((9, 3), dtype=np.float32)
    res = train_tabular_autoencoder(X, ["c1", "c2", "c3"])
    assert res["status"] == "insufficient_data"

    X_empty = np.empty((0, 3), dtype=np.float32)
    res_empty = train_tabular_autoencoder(X_empty, ["c1", "c2", "c3"])
    assert res_empty["status"] == "insufficient_data"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunked Evaluation Mathematical Equivalence & Edge Batch Sizes
# ─────────────────────────────────────────────────────────────────────────────

def test_chunked_evaluation_mathematical_equivalence():
    """Verify chunked evaluation across arbitrary batch sizes yields identical results to unbatched."""
    n_samples = 250
    n_features = 4
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    feature_names = [f"f_{i}" for i in range(n_features)]

    # We instantiate a fixed model to evaluate unbatched vs various chunk sizes
    model = TabularAutoencoderNet(input_dim=n_features, latent_dim=2)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # 1. Ground truth unbatched forward pass
    with torch.no_grad():
        recon_unbatched, latent_unbatched = model(X_tensor)
        expected_recon = recon_unbatched.cpu().numpy()
        expected_latent = latent_unbatched.cpu().numpy()

    # 2. Test various chunk sizes: 1, 7 (prime/odd), 16, 64, 500 (> n_samples)
    for chunk_size in [1, 7, 16, 64, 500]:
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor),
            batch_size=chunk_size,
            shuffle=False,
        )
        recon_chunks = []
        latent_chunks = []
        with torch.no_grad():
            for (bx,) in loader:
                r, l = model(bx)
                recon_chunks.append(r.cpu().numpy())
                latent_chunks.append(l.cpu().numpy())

        actual_recon = np.concatenate(recon_chunks, axis=0)
        actual_latent = np.concatenate(latent_chunks, axis=0)

        # Assert max absolute numerical discrepancy is negligible (< 1e-5)
        max_diff_recon = np.max(np.abs(expected_recon - actual_recon))
        max_diff_latent = np.max(np.abs(expected_latent - actual_latent))
        assert max_diff_recon < 1e-5, f"Recon diverged for chunk_size={chunk_size}: {max_diff_recon}"
        assert max_diff_latent < 1e-5, f"Latent diverged for chunk_size={chunk_size}: {max_diff_latent}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Autoencoder NaN / Inf / -Inf Resilience
# ─────────────────────────────────────────────────────────────────────────────

def test_autoencoder_nan_scattered_and_all_nan_column():
    """Verify handling of scattered NaNs and entirely NaN columns."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 4).astype(np.float32)
    # Column 0 is entirely NaN
    X[:, 0] = np.nan
    # Column 1 has 50% NaNs
    X[:20, 1] = np.nan

    res = train_tabular_autoencoder(X, ["c0", "c1", "c2", "c3"], epochs=2, batch_size=16)
    assert res["status"] == "success"
    assert np.isfinite(res["final_reconstruction_loss"])
    assert not np.isnan(res["final_reconstruction_loss"])
    assert not np.isnan(res["anomaly_threshold_mse"])


def test_autoencoder_inf_and_neginf_adversarial():
    """
    Adversarial Challenge: Verify whether arrays with +inf and -inf produce finite loss.
    Requirement M2-3 explicitly specifies:
    'Verify that arrays with NaNs, infs, or missing values produce finite loss and non-nan outputs.'
    """
    rng = np.random.RandomState(42)
    X = rng.randn(40, 3).astype(np.float32)
    # Inject positive and negative infinities
    X[0, 0] = np.inf
    X[1, 1] = -np.inf
    X[2, 2] = np.inf
    X[10:15, 0] = -np.inf

    res = train_tabular_autoencoder(X, ["a", "b", "c"], epochs=2, batch_size=16)
    assert np.isfinite(res["final_reconstruction_loss"]), f"Expected finite loss, got {res['final_reconstruction_loss']}"
    assert not np.isnan(res["final_reconstruction_loss"]), "Loss is NaN"
    assert not np.isnan(res["anomaly_threshold_mse"]), "Anomaly threshold is NaN"



# ─────────────────────────────────────────────────────────────────────────────
# 4. AutoFE Candidates DataFrame Memory Leak Checks
# ─────────────────────────────────────────────────────────────────────────────

def test_autofe_candidates_df_no_leak_across_multiple_runs():
    """Verify running AutoFE repeatedly does not accumulate dangling DataFrames in heap."""
    gc.collect()
    initial_dfs = sum(1 for obj in gc.get_objects() if isinstance(obj, pd.DataFrame))

    for i in range(10):
        df = pd.DataFrame({
            "num_a": np.random.randn(60),
            "num_b": np.random.randn(60) * 10,
            "target": np.random.choice([0, 1], size=60),
        })
        df_out, feats = auto_engineer_features(df, target_col="target", task_type="classification")
        del df, df_out
        gc.collect()

    final_dfs = sum(1 for obj in gc.get_objects() if isinstance(obj, pd.DataFrame))
    # DataFrame count in heap must not have grown by 10 (which would indicate leak per call)
    growth = final_dfs - initial_dfs
    assert growth < 5, f"Detected potential DataFrame leak: initial={initial_dfs}, final={final_dfs}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Retrieval Model Cache Eviction & Wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieval_cache_eviction_idempotent():
    """Verify clear_retrieval_model_cache is safe and idempotent across multiple consecutive calls."""
    _MODEL_CACHE["m1"] = "dummy"
    _MODEL_CACHE["m2"] = "dummy"
    clear_retrieval_model_cache()
    assert len(_MODEL_CACHE) == 0

    # Second and third calls on empty cache must not raise exceptions
    clear_retrieval_model_cache()
    clear_retrieval_model_cache()
    assert len(_MODEL_CACHE) == 0


def test_retrieval_cache_wiring_with_all_system_components():
    """Verify api.server, dia.session_manager, and BoundedSessionStore all evict retrieval cache."""
    # 1. API server GC
    _MODEL_CACHE["test_key"] = "test_val"
    api_gc()
    assert "test_key" not in _MODEL_CACHE

    # 2. DIA session manager GC
    _MODEL_CACHE["test_key"] = "test_val"
    dia_gc()
    assert "test_key" not in _MODEL_CACHE

    # 3. BoundedSessionStore.clear()
    store = BoundedSessionStore()
    _MODEL_CACHE["test_key"] = "test_val"
    store.clear()
    assert "test_key" not in _MODEL_CACHE
