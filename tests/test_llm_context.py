import pandas as pd
from dia.llm_context import get_dataset_context_and_objectives

def test_get_dataset_context_handles_no_api_keys(monkeypatch):
    df = pd.DataFrame({"Age": [25, 30], "Churn": [0, 1]})
    
    # We expect this to fail gracefully without crashing
    res = get_dataset_context_and_objectives(df)
    
    assert isinstance(res, dict)
    assert "domain" in res
    assert "objectives" in res
