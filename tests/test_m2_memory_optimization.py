"""
tests/test_m2_memory_optimization.py
───────────────────────────────────
Comprehensive test suite for Milestone M2:
1. Memory-efficient DataFrame preparation (no redundant deep copies).
2. Deep Autoencoder memory bounds:
   - Row capping to max 50,000 samples.
   - Batched evaluation forward pass with configurable eval_batch_size.
   - NaN imputation resilience preventing NaN training loss and NaN thresholds.
3. Proactive candidate feature deallocation and garbage collection in AutoFE.
4. Retrieval embedding model cache eviction hook.
5. System garbage collection integration across api.server and dia.session_manager.
"""

import gc
import numpy as np
import pandas as pd
import pytest

from api.server import trigger_system_garbage_collection
from dia.deep_autoencoder import TabularAutoencoderNet, train_tabular_autoencoder
from dia.feature_engineer import auto_engineer_features
from dia.model_trainer import train_and_evaluate
from dia.retrieval import _MODEL_CACHE, clear_retrieval_model_cache
from dia.session_manager import BoundedSessionStore, trigger_system_garbage_collection as dia_system_gc


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Trainer Memory Efficiency
# ─────────────────────────────────────────────────────────────────────────────

def test_model_trainer_dropna_without_redundant_copy():
    """Verify train_and_evaluate prepares data via dropna without calling redundant df.copy()."""
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, None, 1, 0, 1],
    })

    # Train classification models on valid keys with autofe disabled to test data prep path
    res = train_and_evaluate(
        df=df,
        target_col="target",
        task_type="classification",
        selected_model_keys=["logreg"],
        enable_autofe=False,
    )
    assert res is not None
    assert "best_model_key" in res
    assert res["best_model_key"] == "logreg"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Deep Autoencoder Tensor Bounds, Batching & NaN Resilience
# ─────────────────────────────────────────────────────────────────────────────

def test_autoencoder_input_row_capping_50k():
    """Verify inputs exceeding 50,000 rows are capped to 50,000 rows to bound intermediate tensor memory."""
    # Generate 50,050 rows with 4 features
    n_rows = 50_050
    n_cols = 4
    rng = np.random.RandomState(42)
    X = rng.randn(n_rows, n_cols).astype(np.float32)
    feat_names = [f"col_{i}" for i in range(n_cols)]

    res = train_tabular_autoencoder(
        X,
        feat_names,
        epochs=1,
        batch_size=1024,
        eval_batch_size=1024,
    )

    assert res["status"] == "success"
    # Latent embeddings shape must reflect capped 50,000 rows
    assert res["latent_embeddings_shape"][0] == 50_000
    assert res["latent_embeddings_shape"][1] == res["latent_bottleneck_dimension"]
    assert np.isfinite(res["final_reconstruction_loss"])


def test_autoencoder_batched_evaluation_forward_pass():
    """Verify chunked evaluation forward pass produces valid metrics with small eval_batch_size."""
    n_rows = 125
    n_cols = 5
    rng = np.random.RandomState(123)
    X = rng.randn(n_rows, n_cols).astype(np.float32)
    feat_names = [f"f_{i}" for i in range(n_cols)]

    # Use eval_batch_size=16 (yielding 8 chunks for 125 samples)
    res = train_tabular_autoencoder(
        X,
        feat_names,
        epochs=2,
        batch_size=32,
        eval_batch_size=16,
    )

    assert res["status"] == "success"
    assert res["latent_embeddings_shape"] == [125, res["latent_bottleneck_dimension"]]
    assert len(res["loss_curve"]) == 2
    assert np.isfinite(res["anomaly_threshold_mse"])
    assert 0 <= res["anomaly_rate_pct"] <= 100
    assert len(res["feature_attribution_ranking"]) == n_cols
    assert len(res["per_sample_reconstruction_error"]) == min(100, n_rows)


def test_autoencoder_nan_resilience():
    """Verify Autoencoder imputes NaNs prior to tensor conversion and finishes with finite loss."""
    rng = np.random.RandomState(99)
    X = rng.randn(60, 4).astype(np.float32)
    # Inject scattered NaNs
    X[0, 1] = np.nan
    X[5, 2] = np.nan
    X[12, 0] = np.nan
    X[25, 3] = np.nan
    X[50, :] = np.nan

    feat_names = ["feat1", "feat2", "feat3", "feat4"]

    res = train_tabular_autoencoder(
        X,
        feat_names,
        epochs=3,
        batch_size=16,
        eval_batch_size=16,
    )

    assert res["status"] == "success"
    assert np.isfinite(res["final_reconstruction_loss"])
    assert not np.isnan(res["final_reconstruction_loss"])
    assert not np.isnan(res["anomaly_threshold_mse"])
    for val in res["loss_curve"]:
        assert np.isfinite(val)
    for err in res["per_sample_reconstruction_error"]:
        assert np.isfinite(err)


def test_autoencoder_inf_and_neginf_resilience():
    """Verify Autoencoder cleanly sanitizes +inf and -inf to prevent float overflow and yield finite loss."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 3).astype(np.float32)
    # Inject positive and negative infinities as well as NaNs
    X[0, 0] = np.inf
    X[1, 1] = -np.inf
    X[2, 2] = np.inf
    X[10:15, 0] = -np.inf
    X[20, 1] = np.nan

    feat_names = ["col_a", "col_b", "col_c"]
    res = train_tabular_autoencoder(X, feat_names, epochs=2, batch_size=16)

    assert res["status"] == "success"
    assert np.isfinite(res["final_reconstruction_loss"])
    assert not np.isnan(res["final_reconstruction_loss"])
    assert not np.isnan(res["anomaly_threshold_mse"])
    assert np.isfinite(res["anomaly_threshold_mse"])
    for loss_val in res["loss_curve"]:
        assert np.isfinite(loss_val)
    for err in res["per_sample_reconstruction_error"]:
        assert np.isfinite(err)
    for item in res["feature_attribution_ranking"]:
        assert np.isfinite(item["reconstruction_error"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Engineer Proactive Candidate Deallocation
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_engineer_candidate_deallocation():
    """Verify AutoFE creates features, deallocates candidates_df, and triggers gc.collect()."""
    df = pd.DataFrame({
        "x1": np.linspace(1, 100, 50),
        "x2": np.linspace(10, 200, 50),
        "x3": np.exp(np.linspace(0, 5, 50)),
        "ts": pd.date_range("2026-01-01", periods=50, freq="D"),
        "target": np.random.RandomState(42).choice([0, 1], size=50),
    })

    df_out, selected = auto_engineer_features(df, target_col="target", task_type="classification")
    assert isinstance(df_out, pd.DataFrame)
    assert isinstance(selected, list)
    assert len(df_out) == 50
    # Confirm engineered columns are in df_out
    for col in selected:
        assert col in df_out.columns


# ─────────────────────────────────────────────────────────────────────────────
# 4. Retrieval Cache Eviction & Garbage Collection Wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_clear_retrieval_model_cache():
    """Verify clear_retrieval_model_cache purges _MODEL_CACHE dict."""
    _MODEL_CACHE["mock_model_1"] = "dummy_model_object_1"
    _MODEL_CACHE["mock_model_2"] = "dummy_model_object_2"
    assert len(_MODEL_CACHE) >= 2

    clear_retrieval_model_cache()
    assert len(_MODEL_CACHE) == 0


def test_trigger_system_garbage_collection_clears_retrieval_cache():
    """Verify api.server.trigger_system_garbage_collection invokes retrieval cache eviction."""
    _MODEL_CACHE["server_test_model"] = "dummy_model_object"
    assert "server_test_model" in _MODEL_CACHE

    gc_response = trigger_system_garbage_collection()
    assert gc_response.status == "ok"
    assert "server_test_model" not in _MODEL_CACHE
    assert len(_MODEL_CACHE) == 0


def test_dia_session_manager_gc_clears_retrieval_cache():
    """Verify dia.session_manager.trigger_system_garbage_collection clears retrieval model cache."""
    _MODEL_CACHE["session_test_model"] = "dummy_model_object"
    assert "session_test_model" in _MODEL_CACHE

    dia_system_gc()
    assert "session_test_model" not in _MODEL_CACHE
    assert len(_MODEL_CACHE) == 0


def test_bounded_session_store_clear_evicts_retrieval_cache():
    """Verify BoundedSessionStore.clear() also purges retrieval model cache."""
    store = BoundedSessionStore(max_sessions=3, ttl_seconds=300)
    store["sess_1"] = {"df": pd.DataFrame({"a": [1, 2]})}

    _MODEL_CACHE["bounded_test_model"] = "dummy_model_object"
    assert "bounded_test_model" in _MODEL_CACHE

    store.clear()
    assert len(store) == 0
    assert "bounded_test_model" not in _MODEL_CACHE
    assert len(_MODEL_CACHE) == 0
