"""
ui/dashboard.py
───────────────
All Streamlit rendering functions for the Data Intelligence Assistant.
Each function is independent and can be called in sequence from app.py.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from dia.utils import is_plotly_available, is_seaborn_available
from dia.business_metrics import calculate_classification_roi, calculate_regression_impact, generate_actionable_recommendations
from dia.fairness import extract_surrogate_rules, check_disparate_impact


# ─── Colour helpers ───────────────────────────────────────────────────────────

_ROLE_COLOURS = {
    "identifier": "#6366f1",
    "date / time": "#8b5cf6",
    "duration / tenure": "#06b6d4",
    "target candidate": "#f59e0b",
    "categorical feature": "#10b981",
    "numeric feature": "#3b82f6",
    "free text": "#ec4899",
    "binary flag": "#14b8a6",
    "constant (useless)": "#6b7280",
    "high-cardinality categorical": "#f97316",
}

_CONFIDENCE_COLOURS = {
    "High": "🟢",
    "Medium": "🟡",
    "Low": "🔴",
}


def _role_badge(role: str) -> str:
    colour = _ROLE_COLOURS.get(role, "#6b7280")
    safe_role = html.escape(str(role))
    return f'<span style="background:{colour};color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;">{safe_role}</span>'


# ─── 1. Dataset Overview ─────────────────────────────────────────────────────

def render_overview(df: pd.DataFrame, meta: dict) -> None:
    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{meta['n_rows']:,}")
    col2.metric("Columns", f"{meta['n_cols']:,}")
    col3.metric("File Size", f"{meta.get('file_size_mb', 'N/A')} MB")
    col4.metric("Encoding", meta.get("encoding", "utf-8"))

    # Sanitization & Malformed Data Repair Alert
    sanitize_report = meta.get("sanitize_report")
    if sanitize_report and sanitize_report.get("total_cells_repaired", 0) > 0:
        with st.expander(f"🧹 Automated Data Sanitization Report ({sanitize_report['total_cells_repaired']:,} malformed values repaired)", expanded=False):
            from dia.data_sanitizer import format_sanitization_report_markdown
            st.markdown(format_sanitization_report_markdown(sanitize_report))

    # Null heatmap (Plotly) or bar (seaborn fallback)
    null_pcts = (df.isna().mean() * 100).sort_values(ascending=False)
    null_df = null_pcts[null_pcts > 0].reset_index()

    if null_df.empty:
        st.success("✅ No missing values detected in the dataset.")
    else:
        null_df.columns = ["Column", "Missing %"]
        st.markdown("**Missing Value Distribution**")
        if is_plotly_available():
            import plotly.express as px  # type: ignore
            fig = px.bar(
                null_df,
                x="Missing %",
                y="Column",
                orientation="h",
                color="Missing %",
                color_continuous_scale="Reds",
                title="Missing Values per Column",
            )
            fig.update_layout(
                height=max(250, len(null_df) * 28),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        elif is_seaborn_available():
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(8, max(3, len(null_df) * 0.4)))
            sns.barplot(data=null_df, x="Missing %", y="Column", palette="Reds_r", ax=ax)
            ax.set_title("Missing Values per Column")
            st.pyplot(fig)
            plt.close(fig)

    # Data types breakdown
    st.markdown("**Column Data Types**")
    dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ["dtype", "count"]
    if is_plotly_available():
        import plotly.express as px
        fig = px.pie(
            dtype_counts,
            names="dtype",
            values="count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Preview table
    with st.expander("🔍 Preview first 10 rows", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)


# ─── 2. Inferred Column Roles ─────────────────────────────────────────────────

def render_column_roles(annotated_profile: pd.DataFrame) -> None:
    st.subheader("🔍 Inferred Column Roles")

    st.markdown(
        "The assistant has analysed each column and assigned an inferred role "
        "based on its name, data type, and value distribution."
    )

    display_cols = ["column", "dtype", "null_pct", "unique_count",
                    "inferred_role", "confidence_label", "explanation"]
    present = [c for c in display_cols if c in annotated_profile.columns]
    df_show = annotated_profile[present].copy()
    df_show.columns = [c.replace("_", " ").title() for c in present]

    # Confidence with emoji
    if "Confidence Label" in df_show.columns:
        df_show["Confidence Label"] = df_show["Confidence Label"].apply(
            lambda x: f"{_CONFIDENCE_COLOURS.get(x, '')} {x}"
        )

    st.dataframe(df_show, use_container_width=True, height=350)

    # Role frequency chart
    role_counts = annotated_profile["inferred_role"].value_counts().reset_index()
    role_counts.columns = ["Role", "Count"]

    if is_plotly_available():
        import plotly.express as px
        fig = px.bar(
            role_counts,
            x="Count",
            y="Role",
            orientation="h",
            color="Role",
            color_discrete_sequence=list(_ROLE_COLOURS.values()),
            title="Column Role Distribution",
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10),
            height=max(200, len(role_counts) * 32),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── 3. Data Readiness Report ─────────────────────────────────────────────────

def render_readiness_report(readiness: dict, goal_info: dict, target_col: str, task_type: str) -> None:
    st.subheader("🩺 Data Readiness Report")

    # Verdict banner
    verdict = readiness["verdict"]
    score = readiness["score"]

    if "✅" in verdict:
        st.success(f"**{verdict}** — Readiness Score: {score}/100")
    elif "⚠️" in verdict:
        st.warning(f"**{verdict}** — Readiness Score: {score}/100")
    else:
        st.error(f"**{verdict}** — Readiness Score: {score}/100")

    # Score gauge
    if is_plotly_available():
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 44], "color": "#fca5a5"},
                    {"range": [45, 74], "color": "#fde68a"},
                    {"range": [75, 100], "color": "#6ee7b7"},
                ],
            },
            title={"text": "Readiness Score"},
        ))
        fig.update_layout(
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Goal parse result
    st.markdown(f"**🎯 Goal Interpretation:** {goal_info['explanation']}")
    st.markdown(f"**📌 Selected Target Column:** `{target_col}`")
    st.markdown(f"**🧪 Task Type:** `{task_type}`")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**✅ Useful Features**")
        for f in readiness["useful_features"] or ["None detected"]:
            st.markdown(f"- `{f}`")
    with c2:
        st.markdown("**⚠️ Possible Leakage Risk**")
        for f in readiness["leakage_risk"] or ["None detected"]:
            st.markdown(f"- `{f}`")
    with c3:
        st.markdown("**❓ Missing Signals**")
        for f in readiness["missing_signals"] or ["None noted"]:
            st.markdown(f"- {f}")

    if readiness["summary_lines"]:
        st.markdown("**📋 Additional Notes**")
        for line in readiness["summary_lines"]:
            st.markdown(f"- {line}")


# ─── 4. Model Results ─────────────────────────────────────────────────────────

def render_model_results(train_result: dict) -> None:
    st.subheader("🤖 Model Training Results")

    results = train_result["results"]
    best_key = train_result["best_model_key"]
    task_type = train_result["task_type"]
    justification = train_result.get("justification", "")

    st.success(f"🏆 Best Model: **{train_result['best_model_label']}**")
    if justification:
        st.info(justification)

    # Metrics comparison table
    metric_rows = []
    for r in results:
        if r.get("error"):
            metric_rows.append({"Model": r["label"], **{"Error": r["error"]}})
        else:
            row = {"Model": r["label"]}
            row.update(r["metrics"])
            if r["model_key"] == best_key:
                row["Model"] = f"⭐ {row['Model']}"
            metric_rows.append(row)

    if metric_rows:
        metrics_df = pd.DataFrame(metric_rows).set_index("Model")
        st.dataframe(
            metrics_df.style.format(
                {col: "{:.4f}" for col in metrics_df.select_dtypes("number").columns}
            ).highlight_max(axis=0, color="#6ee7b7"),
            use_container_width=True,
        )

    # Radar / bar chart comparing models
    valid_results = [r for r in results if r.get("metrics")]
    if len(valid_results) > 1 and is_plotly_available():
        import plotly.graph_objects as go
        import plotly.express as px

        metric_keys = list(valid_results[0]["metrics"].keys())

        fig = go.Figure()
        for r in valid_results:
            vals = [r["metrics"].get(k, 0) for k in metric_keys]
            name = f"⭐ {r['label']}" if r["model_key"] == best_key else r["label"]
            fig.add_trace(go.Bar(name=name, x=metric_keys, y=vals))

        fig.update_layout(
            barmode="group",
            title="Model Comparison",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif len(valid_results) > 1 and is_seaborn_available():
        import matplotlib.pyplot as plt
        import seaborn as sns
        metric_keys = list(valid_results[0]["metrics"].keys())
        rows = []
        for r in valid_results:
            for k in metric_keys:
                rows.append({"Model": r["label"], "Metric": k, "Score": r["metrics"].get(k, 0)})
        plot_df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", ax=ax)
        ax.set_title("Model Comparison")
        st.pyplot(fig)
        plt.close(fig)

    # ── Advanced Evaluation Plots ─────────────────────────────────────────────
    st.markdown("#### 📈 Best Model Evaluation Details")
    
    if "y_true" in train_result and "y_pred" in train_result and is_plotly_available():
        
        y_true = train_result["y_true"]
        y_pred = train_result["y_pred"]
        
        if task_type == "classification":
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Use label names if available
            le = train_result.get("label_encoder")
            if le is not None:
                labels = le.classes_.astype(str)
            else:
                labels = [str(i) for i in range(cm.shape[0])]
                
            fig = px.imshow(
                cm, 
                text_auto=True, 
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                x=labels, y=labels,
                title="Confusion Matrix"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10, l=10, r=10), height=350)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Regression: Residual plot
            residuals = y_true - y_pred
            df_res = pd.DataFrame({"Predicted": y_pred, "Residuals": residuals})
            
            fig = px.scatter(
                df_res, x="Predicted", y="Residuals", 
                title="Residuals vs Predicted",
                opacity=0.6,
                color_discrete_sequence=["#3b82f6"]
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10, l=10, r=10), height=350)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tuned Hyperparameters ────────────────────────────────────────────────
    models_with_params = [r for r in results if r.get("best_params")]
    if models_with_params:
        with st.expander("⚙️ Tuned Hyperparameters (HPO)"):
            for r in models_with_params:
                st.markdown(f"**{r['label']}**")
                param_df = pd.DataFrame(
                    [{"Hyperparameter": k, "Tuned Value": str(v)} for k, v in r["best_params"].items()]
                )
                st.table(param_df)


# ─── 4.5. Smart Insights ───────────────────────────────────────────────────────

def render_smart_insights(insights: list[dict]) -> None:
    st.subheader("💡 Automated Smart Insights")
    st.markdown("The assistant has analyzed relationships between your features and the target variable to surface these key insights:")
    
    for item in insights:
        type_color = {
            "positive": "border-left: 4px solid #10b981;",
            "negative": "border-left: 4px solid #ef4444;",
            "neutral": "border-left: 4px solid #6366f1;"
        }.get(item["type"], "")
        
        icon = {
            "positive": "📈",
            "negative": "📉",
            "neutral": "🔍"
        }.get(item["type"], "💡")
        
        safe_title = html.escape(str(item.get("title", "")))
        safe_desc = html.escape(str(item.get("description", "")))
        
        st.markdown(f"""
        <div style="background: var(--bg-card); padding: 1rem; border-radius: var(--radius); margin-bottom: 1rem; {type_color}">
            <h4 style="margin-top: 0;">{icon} {safe_title}</h4>
            <p style="margin-bottom: 0;">{safe_desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ─── 5. Explainability ───────────────────────────────────────────────────────

def render_explainability(explanation: dict, best_label: str) -> None:
    st.subheader("🔬 Feature Importance & Explainability")

    method = explanation.get("method", "none")
    figure = explanation.get("figure")

    if method == "none" or figure is None:
        st.info("Feature importance could not be generated for the selected model.")
        return

    method_labels = {
        "shap": "SHAP (SHapley Additive exPlanations)",
        "plotly_importance": "Model Feature Importances",
        "seaborn_importance": "Model Feature Importances (static)",
    }
    st.caption(f"Method: **{method_labels.get(method, method)}** — Model: **{best_label}**")

    if method in ("shap", "plotly_importance"):
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.pyplot(figure)


# ─── 6. Final Summary ─────────────────────────────────────────────────────────

def render_final_summary(
    df: pd.DataFrame,
    goal: str,
    goal_info: dict,
    target_col: str,
    task_type: str,
    readiness: dict,
    train_result: dict,
    annotated_profile: pd.DataFrame,
) -> None:
    st.subheader("📝 Final Plain-English Summary")

    best_label = train_result["best_model_label"]
    best_metrics = train_result["best_metrics"]
    verdict = readiness["verdict"]
    n_rows, n_cols = df.shape
    n_useful = len(readiness["useful_features"])
    n_leakage = len(readiness["leakage_risk"])

    # Format best metric for the summary
    if task_type == "classification":
        primary_metric_name = "ROC-AUC" if "ROC-AUC" in best_metrics else "F1"
        primary_metric_val = best_metrics.get(primary_metric_name, "N/A")
    else:
        primary_metric_name = "R²"
        primary_metric_val = best_metrics.get("R²", "N/A")

    target_role_row = annotated_profile[annotated_profile["column"] == target_col]
    target_role = (
        target_role_row["inferred_role"].iloc[0]
        if not target_role_row.empty
        else "unknown"
    )

    summary = f"""
### What the assistant understood
You asked: **"{goal}"**

The assistant parsed this as a **{task_type}** problem with confidence **{goal_info['confidence']:.0%}** using the **{goal_info['method']}** matching strategy.

### What was inferred about the dataset
- The dataset has **{n_rows:,} rows** and **{n_cols:,} columns**.
- The target column **`{target_col}`** was identified (inferred role: *{target_role}*).
- **{n_useful}** likely useful feature(s) were detected for model training.
- **{n_leakage}** column(s) were flagged as potential data leakage risks (identifiers, raw dates, etc.).

### Why this task type was chosen
{"The target column contains discrete categories or a binary outcome, making it a classification task." if task_type == "classification" else "The target column contains continuous numeric values, making it a regression task."}

### Which model performed best
🏆 **{best_label}** achieved a **{primary_metric_name}** of **{primary_metric_val}**, outperforming the other trained models.

### Data trustworthiness verdict
{verdict.replace("✅ ", "").replace("⚠️ ", "").replace("❌ ", "")} (Readiness Score: **{readiness['score']}/100**)

{"The data appears suitable for your stated goal. Treat model results as a baseline — real-world deployment requires additional validation." if readiness['score'] >= 75 else "There are notable data quality issues. Consider cleaning missing values, removing leakage-risk columns, and collecting more data before trusting these results in production."}
"""

    st.markdown(summary)

    # Export summary as text
    st.download_button(
        label="⬇️ Download Summary as Text",
        data=summary,
        file_name="dia_summary.md",
        mime="text/markdown",
    )


# ─── 7. Business Impact & Fairness ────────────────────────────────────────────

def render_business_impact(train_result: dict, df: pd.DataFrame, target_col: str, task_type: str, insights: list[dict]):
    """Renders the business impact, ROI simulator, and fairness metrics."""
    st.subheader("💼 Business Impact & ROI Simulator")
    
    # Archetype selector
    st.markdown("Select your primary business objective to tailor the financial simulator and recommendations:")
    archetype = st.selectbox(
        "Business Objective",
        options=["custom", "retention", "lead", "fraud"],
        format_func=lambda x: {
            "custom": "Custom ROI / Generic",
            "retention": "Customer Retention & Churn",
            "lead": "Lead Scoring & Conversion",
            "fraud": "Risk & Fraud Detection"
        }[x]
    )
    
    if task_type == "classification":
        st.markdown(
            "Simulate the financial impact of deploying this model. "
            "Adjust the parameters below to see how the model affects your bottom line compared to a baseline."
        )
        col1, col2 = st.columns(2)
        
        if archetype == "retention":
            with col1:
                ltv = st.number_input("Average Customer Lifetime Value ($)", min_value=0.0, value=1000.0, step=100.0)
                offer_cost = st.number_input("Cost of Retention Offer ($)", min_value=0.0, value=100.0, step=10.0)
            cost_fp, cost_fn, val_tp, val_tn = offer_cost, ltv, ltv - offer_cost, 0.0
        elif archetype == "lead":
            with col1:
                deal_size = st.number_input("Average Deal Size ($)", min_value=0.0, value=5000.0, step=500.0)
                contact_cost = st.number_input("Cost per Sales Outreach ($)", min_value=0.0, value=50.0, step=10.0)
            cost_fp, cost_fn, val_tp, val_tn = contact_cost, deal_size, deal_size - contact_cost, 0.0
        elif archetype == "fraud":
            with col1:
                txn_val = st.number_input("Average Transaction Value ($)", min_value=0.0, value=200.0, step=20.0)
                investigation_cost = st.number_input("Cost of Manual Review ($)", min_value=0.0, value=25.0, step=5.0)
            cost_fp, cost_fn, val_tp, val_tn = investigation_cost, txn_val, txn_val - investigation_cost, 0.0
        else:
            with col1:
                cost_fp = st.number_input("Cost of a False Positive ($)", min_value=0.0, value=50.0, step=10.0, 
                                          help="E.g., money wasted targeting someone who won't convert.")
                cost_fn = st.number_input("Cost of a False Negative ($)", min_value=0.0, value=500.0, step=50.0,
                                          help="E.g., revenue lost from missing a churner.")
            with col2:
                val_tp = st.number_input("Value of a True Positive ($)", min_value=0.0, value=200.0, step=20.0,
                                         help="E.g., revenue gained from a successful intervention.")
                val_tn = st.number_input("Value of a True Negative ($)", min_value=0.0, value=0.0, step=10.0)
                
        roi_data = calculate_classification_roi(
            train_result["y_true"], train_result["y_pred"],
            cost_fp, cost_fn, val_tp, val_tn
        )
        
        if "error" in roi_data:
            st.warning(roi_data["error"])
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Model Net ROI", f"${roi_data['net_roi']:,.2f}", 
                      delta=f"${roi_data['model_savings']:,.2f} vs Baseline")
            m2.metric("Total Value Created", f"${roi_data['total_value']:,.2f}")
            m3.metric("Total Error Cost", f"${roi_data['total_cost']:,.2f}", delta_color="inverse")
            
    else:
        st.markdown("Estimate the financial impact of prediction errors.")
        avg_target = df[target_col].mean() if pd.api.types.is_numeric_dtype(df[target_col]) else 0
        mae = train_result["best_metrics"].get("MAE", 0)
        n_preds = st.number_input("Number of Predictions per Month", min_value=1, value=10000, step=1000)
        
        impact = calculate_regression_impact(mae, n_preds, avg_target)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Average Error per Prediction", f"{impact['avg_error_per_prediction']:,.2f}")
        m2.metric("Total Monthly Error Cost", f"{impact['total_error']:,.2f}", delta_color="inverse")
        m3.metric("Error Margin", f"{impact['error_margin_pct']:.1f}% of Average Target")

    st.divider()
    
    col_rec, col_fair = st.columns(2)
    
    with col_rec:
        st.subheader("🎯 Actionable Recommendations")
        best_imp = train_result.get("best_importance")
        if best_imp is None or best_imp.empty:
            best_imp = train_result["results"][0].get("importance", pd.Series(dtype=float))
        actions = generate_actionable_recommendations(
            best_imp,
            insights,
            archetype=archetype
        )
        for act in actions:
            st.info(act)
            
        st.subheader("👤 Ideal Profile Rules")
        if task_type == "classification":
            rules = extract_surrogate_rules(
                train_result["X_test_processed"], 
                train_result["y_pred"], 
                train_result["feature_names"]
            )
            with st.expander("View Surrogate Decision Tree Rules"):
                st.code(rules, language="text")
        else:
            st.write("Profile rules are primarily supported for classification tasks.")
            
    with col_fair:
        st.subheader("⚖️ Fairness & Risk Scan")
        st.markdown("Scans categorical groups for disparate impact (e.g. >15% accuracy drop).")
        
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_col in cat_cols:
            cat_cols.remove(target_col)
            
        try:
            X_all = train_result["preprocessor"].transform(df.drop(columns=[target_col]))
            y_all_pred = train_result["best_model"].predict(X_all)
            
            if task_type == "classification":
                y_all_true = train_result["label_encoder"].transform(df[target_col].astype(str))
            else:
                y_all_true = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values
                
            alerts = check_disparate_impact(df, y_all_true, y_all_pred, cat_cols)
            
            if not alerts:
                st.success("✅ No significant disparate impact detected across major categorical groups.")
            else:
                for alert in alerts:
                    st.warning(alert["alert"])
        except Exception as e:
            st.caption(f"Fairness scan could not be completed: {e}")
