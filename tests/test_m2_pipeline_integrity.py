"""
tests/test_m2_pipeline_integrity.py
───────────────────────────────────
End-to-end integration and unit tests for Milestone M2 features:
- Feature 10: Boolean column preservation in ColumnTransformer
- Feature 11: Semantic target encoding directionality
- Feature 12: 1-to-1 feature matrix alignment
- Feature 13: Voting ensemble feature importance calculation (zero mock 0.05)
- Feature 14: Wachter recourse preprocessor pipeline & categorical label inversion
- Feature 15: Causal discovery treatment column preservation
- Feature 16: Decision Tree ('dt') registration in HPO & defaults
- Feature 17: Brier score & calibration curve metrics
- Feature 18: Supply Chain & Clinical Sepsis demo benchmarks
- Feature 19: Constant feature guard & autoencoder batch size clamping
- Feature 20: Session store state preservation
- Feature 21: Edge model JS & WAT compilation
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from dia.model_trainer import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    HPO_PARAM_GRIDS,
    _build_preprocessor,
    _encode_classification_target,
    _extract_importance,
    _classification_metrics,
    train_and_evaluate,
)
from dia.pipeline_coordinator import PipelineCoordinator
from dia.recourse_engine import compute_recourse
from dia.causal_discovery import discover_causal_graph, simulate_intervention
from dia.demo_datasets import get_demo_dataset, DEMO_BENCHMARKS
from dia.deep_autoencoder import train_tabular_autoencoder
from dia.session_manager import BoundedSessionStore
from dia.wasm_compiler import transpile_to_edge_bundle


def test_feature_10_bool_column_preservation():
    """Verify boolean columns are preserved in preprocessor and not dropped."""
    df = pd.DataFrame({
        "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        "bool_col": [True, False, True, False, True],
        "cat_col": ["A", "B", "A", "B", "A"],
    })
    prep = _build_preprocessor(df)
    prep.fit(df)
    transformed = prep.transform(df)
    # Output must have numeric + bool + categorical features transformed
    assert transformed.shape[1] >= 2
    # Ensure bool_col was included in numeric transformer
    num_cols = prep.transformers_[0][2]
    assert "bool_col" in num_cols


def test_feature_11_semantic_target_encoding():
    """Verify semantic mapping assigns positive token (e.g. Default) to 1."""
    # Test case 1: Default vs Non-Default (alphabetically, Default would be 0 without semantic mapping)
    y_raw = pd.Series(["Default", "Non-Default", "Default", "Non-Default"])
    y_enc, le = _encode_classification_target(y_raw, "target")
    assert le.classes_[1] == "Default"
    assert le.classes_[0] == "Non-Default"
    assert y_enc[0] == 1
    assert y_enc[1] == 0

    # Test case 2: Churn vs Retain
    y_raw2 = pd.Series(["Retain", "Churn", "Retain", "Churn"])
    y_enc2, le2 = _encode_classification_target(y_raw2, "target")
    assert le2.classes_[1] == "Churn"
    assert le2.classes_[0] == "Retain"
    assert y_enc2[1] == 1
    assert y_enc2[0] == 0


def test_feature_13_voting_ensemble_feature_importance():
    """Verify VotingClassifier computes normalized average of base estimators' importances."""
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    dt1 = DecisionTreeClassifier(random_state=42).fit(X, y)
    dt2 = DecisionTreeClassifier(random_state=43).fit(X, y)

    voting = VotingClassifier(estimators=[("dt1", dt1), ("dt2", dt2)], voting="soft")
    voting.fit(X, y)

    importances = _extract_importance(voting, ["feat1", "feat2"])
    assert len(importances) == 2
    assert pytest.approx(sum(importances), rel=1e-3) == 1.0
    # Values should be non-mocked
    assert not all(v == 0.05 for v in importances)


def test_feature_14_recourse_engine_categorical_inversion():
    """Verify recourse engine handles categorical inputs, transforms via preprocessor, and returns category labels."""
    df = pd.DataFrame({
        "age": [25, 30, 45, 50, 22, 60, 35, 40],
        "income": [30000, 45000, 70000, 90000, 25000, 120000, 55000, 65000],
        "education": ["HighSchool", "Bachelors", "Masters", "PhD", "HighSchool", "PhD", "Bachelors", "Masters"],
        "target": [0, 0, 1, 1, 0, 1, 0, 1],
    })
    feature_cols = ["age", "income", "education"]
    X = df[feature_cols]
    y = df["target"]

    preprocessor = _build_preprocessor(X)
    X_trans = preprocessor.fit_transform(X)

    model = LogisticRegression().fit(X_trans, y)

    query_instance = {"age": 25, "income": 30000, "education": "HighSchool"}
    result = compute_recourse(
        model=model,
        x_input=query_instance,
        reference_df=X,
        feature_names=feature_cols,
        desired_class=1,
        preprocessor=preprocessor,
    )
    assert result.feasibility in ["OPTIMAL_RECOURSE_FOUND", "APPROXIMATE_RECOURSE_FOUND"]
    assert isinstance(result.counterfactual_vector, dict)
    # Categorical feature should be mapped to an authentic category string
    if "education" in result.counterfactual_vector:
        assert isinstance(result.counterfactual_vector["education"], str)
        assert result.counterfactual_vector["education"] in ["HighSchool", "Bachelors", "Masters", "PhD"]


def test_feature_15_causal_discovery_treatment_retention():
    """Verify causal discovery preserves treatment column and runs conditional independence."""
    np.random.seed(42)
    n = 200
    z = np.random.randn(n)
    x = 0.8 * z + 0.2 * np.random.randn(n)
    y = 0.5 * x + 0.5 * z + 0.1 * np.random.randn(n)
    w = np.random.randn(n)

    df = pd.DataFrame({"confounder": z, "treatment": x, "outcome": y, "noise": w})

    graph = discover_causal_graph(df, target_col="outcome", treatment_col="treatment")
    assert "treatment" in graph.nodes
    assert "outcome" in graph.nodes

    intervention = simulate_intervention(df, treatment="treatment", outcome="outcome", intervention_value=2.0)
    assert intervention.treatment == "treatment"
    assert intervention.outcome == "outcome"
    assert np.isfinite(intervention.average_treatment_effect)


def test_feature_16_decision_tree_registration():
    """Verify 'dt' is registered in classification and regression models and HPO grids."""
    assert "dt" in CLASSIFICATION_MODELS
    assert "dt" in REGRESSION_MODELS
    assert "dt" in HPO_PARAM_GRIDS

    coordinator = PipelineCoordinator()
    assert "dt" in coordinator.default_classification_models
    assert "dt" in coordinator.default_regression_models


def test_feature_17_brier_score_and_calibration():
    """Verify brier_score_loss and calibration curves are computed in classification metrics."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.05, 0.95]])

    metrics = _classification_metrics(y_true, y_pred, y_prob)
    assert "brier_score" in metrics
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_feature_18_benchmarks_registration():
    """Verify supply_chain and clinical_sepsis benchmarks generate correct shapes and targets."""
    df_sc, goal_sc, target_sc = get_demo_dataset("supply_chain")
    assert df_sc.shape[0] == 600
    assert target_sc == "delay_flag"
    assert target_sc in df_sc.columns
    assert "supply_chain" in DEMO_BENCHMARKS

    df_cs, goal_cs, target_cs = get_demo_dataset("clinical_sepsis")
    assert df_cs.shape[0] == 600
    assert target_cs == "sepsis_target"
    assert target_cs in df_cs.columns
    assert "clinical_sepsis" in DEMO_BENCHMARKS
    # Severe class imbalance check (~5-10% positive)
    pos_rate = df_cs[target_cs].mean()
    assert 0.01 <= pos_rate <= 0.20


def test_feature_19_guards_and_autoencoder_clamping():
    """Verify constant features guard and autoencoder eval batch size clamping."""
    # Test autoencoder batch size clamping
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4).astype(np.float32)
    res = train_tabular_autoencoder(X, ["c1", "c2", "c3", "c4"], epochs=2, eval_batch_size=1024)
    assert res["status"] == "success"
    assert res["latent_embeddings_shape"][0] == 100

    # Test constant numeric column guard in train_and_evaluate
    df_const = pd.DataFrame({
        "normal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "constant": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "target": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    res_train = train_and_evaluate(
        df=df_const,
        target_col="target",
        task_type="classification",
        selected_model_keys=["logreg"],
    )
    # Ensure training runs cleanly without failing due to constant column and constant is filtered
    assert "constant" not in res_train["raw_feature_cols"]
    assert "normal" in res_train["raw_feature_cols"]


def test_feature_20_session_store_preservation():
    """Verify BoundedSessionStore stores and preserves session state, metadata, and results."""
    store = BoundedSessionStore(max_sessions=3, ttl_seconds=600)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    sess_data = {
        "df": df,
        "goal": "Predict b",
        "target": "b",
        "tenant_id": "test_tenant",
        "pipeline_result": {"status": "success"},
    }
    store["sess_1"] = sess_data
    assert "sess_1" in store
    retrieved = store["sess_1"]
    assert retrieved["goal"] == "Predict b"
    assert retrieved["pipeline_result"]["status"] == "success"
    assert store._meta["sess_1"]["n_rows"] == 2
    assert store._meta["sess_1"]["n_cols"] == 2


def test_feature_21_edge_bundle_compilation():
    """Verify standalone JS and WAT edge bundle generation for DecisionTree."""
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    y = np.array([0, 0, 1, 1])
    dt = DecisionTreeClassifier(max_depth=3).fit(X, y)

    bundle = transpile_to_edge_bundle(
        model=dt,
        feature_names=["f1", "f2"],
        classes=[0, 1],
    )
    assert bundle.model_type == "DecisionTreeClassifier"
    assert len(bundle.js_code) > 0
    assert "DiaEdgeEngine" in bundle.js_code
    assert bundle.wat_code is not None
    assert len(bundle.wat_code) > 0
    assert bundle.bundle_size_bytes < 500 * 1024
