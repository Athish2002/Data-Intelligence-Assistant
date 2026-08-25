"""Quick end-to-end backend test — run with: python test_e2e.py"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from dia.goal_parser import parse_goal
from dia.data_profiler import (
    profile_dataframe, infer_column_roles,
    detect_target_type, generate_readiness_report,
)
from dia.model_trainer import train_and_evaluate

np.random.seed(42)
n = 300
df = pd.DataFrame({
    "customer_id":     range(n),
    "tenure_months":   np.random.randint(1, 72, n),
    "monthly_charges": np.random.uniform(20, 120, n),
    "num_services":    np.random.randint(1, 8, n),
    "joined_on":       pd.date_range("2018-01-01", periods=n, freq="D").astype(str),
    "churn_flag":      np.random.choice([0, 1], n, p=[0.75, 0.25]),
})

# 1. Goal parsing
gi = parse_goal("predict customer churn", columns=df.columns.tolist())
print(f"[goal]   task={gi['task_type']}  conf={gi['confidence']:.0%}  candidates={gi['target_candidates'][:3]}")

# 2. Profile + role inference
profile = profile_dataframe(df)
annotated = infer_column_roles(df, profile)
for _, row in annotated.iterrows():
    print(f"  {row['column']:<20} -> {row['inferred_role']}  ({row['confidence_label']})")

# 3. Target type detection
td = detect_target_type(df, "churn_flag")
print(f"[target] type={td['task_type']}  reason={td['reason']}")

# 4. Readiness report
r = generate_readiness_report(df, annotated, gi, "churn_flag", "classification")
print(f"[ready]  score={r['score']}  verdict={r['verdict'].replace(chr(9989)+' ','').replace(chr(9888)+chr(65039)+' ','').replace(chr(10060)+' ','')}")
print(f"         useful={r['useful_features'][:3]}  leakage={r['leakage_risk'][:3]}")

# 5. Train
result = train_and_evaluate(df, "churn_flag", "classification", ["logreg", "rf"])
print(f"[model]  best={result['best_model_label']}")
for k, v in result["best_metrics"].items():
    print(f"         {k}: {v}")

print("\nALL TESTS PASSED")
