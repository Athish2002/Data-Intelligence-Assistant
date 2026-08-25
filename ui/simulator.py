"""
ui/simulator.py
───────────────
Interactive What-If Simulator. Auto-generates UI widgets based on feature metadata
and runs live predictions using the trained model and preprocessor.
"""

import pandas as pd
import streamlit as st

def render_simulator(df: pd.DataFrame, feature_names: list[str], preprocessor, model, task_type: str, label_encoder=None):
    st.markdown("### 🎛️ Interactive What-If Simulator")
    st.write("Tweak the feature values below to see how the model's prediction changes in real-time.")
    
    # ── 1. Create default input data based on medians/modes
    defaults = {}
    for col in feature_names:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                defaults[col] = float(df[col].median())
            else:
                defaults[col] = df[col].mode().iloc[0]
                
    st.markdown("#### Input Features")
    
    # ── 2. Render dynamic form inputs
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    user_inputs = {}
    
    for i, col in enumerate(feature_names):
        if col not in df.columns:
            continue
            
        with cols[i % 3]:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                # Handle cases where min == max
                if min_val == max_val:
                    min_val = min_val - 1.0
                    max_val = max_val + 1.0
                
                step = (max_val - min_val) / 100.0
                if step == 0:
                    step = 0.1
                    
                user_inputs[col] = st.slider(
                    f"`{col}`",
                    min_value=min_val,
                    max_value=max_val,
                    value=defaults[col],
                    step=step,
                    key=f"sim_{col}"
                )
            else:
                options = df[col].dropna().unique().tolist()
                user_inputs[col] = st.selectbox(
                    f"`{col}`",
                    options=options,
                    index=options.index(defaults[col]) if defaults[col] in options else 0,
                    key=f"sim_{col}"
                )
                
    st.divider()
    
    # ── 3. Run prediction
    st.markdown("#### 🔮 Live Prediction")
    
    # Convert user inputs to DataFrame matching original column order
    input_df = pd.DataFrame([user_inputs])
    
    try:
        # Preprocess
        X_proc = preprocessor.transform(input_df)
        
        # Predict
        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_proc)[0]
                pred_class_idx = proba.argmax()
                pred_class_name = label_encoder.inverse_transform([pred_class_idx])[0] if label_encoder else str(pred_class_idx)
                
                st.metric(label="Predicted Class", value=str(pred_class_name))
                
                st.write("**Class Probabilities:**")
                # Ensure proba is an iterable of floats, not a 0-d array
                import numpy as np
                proba_list = proba.flatten().tolist() if isinstance(proba, np.ndarray) else list(proba)
                
                prob_data = {}
                for idx, p in enumerate(proba_list):
                    c_name = label_encoder.inverse_transform([idx])[0] if label_encoder else str(idx)
                    prob_data[c_name] = p
                    
                st.bar_chart(pd.Series(prob_data))
                
            else:
                pred = model.predict(X_proc)[0]
                pred_name = label_encoder.inverse_transform([pred])[0] if label_encoder else str(pred)
                st.metric(label="Predicted Class", value=str(pred_name))
                
        else:
            pred = model.predict(X_proc)[0]
            st.metric(label="Predicted Value", value=f"{pred:,.2f}")
            
    except Exception as e:
        st.error(f"Could not generate prediction: {str(e)}")
