import pandas as pd
from dia.insights import generate_smart_insights

def test_generate_smart_insights_regression():
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5] * 5,
        "feature1": [2, 4, 6, 8, 10] * 5, 
        "feature2": [10, 8, 6, 4, 2] * 5,
        "feature3": [1, 1, 1, 1, 1] * 5
    })
    
    insights = generate_smart_insights(df, "target", "regression")
    assert len(insights) >= 2
    
    titles = [i["title"] for i in insights]
    assert any("feature1" in t for t in titles)
    assert any("feature2" in t for t in titles)
    
def test_generate_smart_insights_classification():
    df = pd.DataFrame({
        "target": ["A", "B", "A", "B", "B"] * 5,
        "feature1": [1, 10, 2, 9, 11] * 5, 
        "feature2": [5, 5, 5, 5, 5] * 5
    })
    
    insights = generate_smart_insights(df, "target", "classification")
    assert len(insights) >= 1
    
    titles = [i["title"] for i in insights]
    assert any("feature1" in t for t in titles)

def test_generate_smart_insights_fallback():
    # Only categorical features, no numeric correlations possible
    df = pd.DataFrame({
        "target": [1, 2, 3],
        "feature1": ["a", "b", "c"]
    })
    insights = generate_smart_insights(df, "target", "regression")
    assert len(insights) == 1
    assert "Complex Relationships" in insights[0]["title"]
