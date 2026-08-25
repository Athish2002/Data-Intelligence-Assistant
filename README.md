# 🧠 Data Intelligence Assistant

> An intelligent data science assistant that helps you upload a CSV dataset, describe your machine-learning goal in plain English, and receive a full diagnostic report plus baseline model training — all inside a clean, modern web interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Smart Dataset Profiling** | Automatically detects column types, null percentages, unique counts, high-cardinality columns, and likely identifier / date columns |
| **Intelligent Column Role Inference** | Uses heuristics to label each column as identifier, date, duration, target candidate, numeric feature, etc. — with a confidence score |
| **Dual Goal Parsing** | Maps your free-text goal to a task type using both **keyword matching** and optional **semantic matching** (sentence-transformers) |
| **Automatic Task Detection** | Detects classification vs. regression from both the goal text and the target column's data type and unique-value ratio |
| **Data Readiness Report** | Scores your dataset 0–100 for suitability, highlights useful features, flags leakage risks, and lists missing signals |
| **Model Selection** | Choose up to 4 models before training: Logistic Regression, Random Forest, XGBoost (optional), LightGBM (optional) |
| **Baseline Model Training** | Trains selected models with proper preprocessing (imputation, scaling, one-hot encoding) and an 80/20 train-test split |
| **Full Evaluation Metrics** | Classification: Accuracy, Precision, Recall, F1, ROC-AUC · Regression: MAE, RMSE, R² |
| **Feature Explainability** | SHAP values if available, falling back to sklearn feature importances, rendered as Plotly charts or Seaborn static plots |
| **Plain-English Final Summary** | A human-readable report summarising what the assistant understood, what it inferred, which model won, and whether the data is trustworthy |
| **Downloadable Summary** | Export the final summary as a Markdown file |
| **Dark Mode Ready** | Toggle via Streamlit's built-in Settings menu |
| **500 MB Upload Limit** | Files are kept in memory only — nothing is saved to disk |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/data-intelligence-assistant.git
cd data-intelligence-assistant
```

### 2. Create a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Optional heavy dependencies** (comment out in `requirements.txt` if not needed):
> - `shap` — for SHAP-based explainability
> - `sentence-transformers` — for semantic goal parsing
> - `xgboost` — for XGBoost model option
> - `lightgbm` — for LightGBM model option

### 4. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
data-intelligence-assistant/
│
├── app.py                  # Streamlit entry point
│
├── dia/                    # Backend logic package
│   ├── __init__.py
│   ├── data_loader.py      # CSV loading with 500 MB size check
│   ├── goal_parser.py      # Keyword + semantic goal parsing
│   ├── data_profiler.py    # Column profiling, role inference, readiness report
│   ├── model_trainer.py    # Multi-model training and evaluation
│   ├── explainability.py   # SHAP / Plotly / Seaborn importance charts
│   └── utils.py            # Shared helpers and constants
│
├── ui/                     # Streamlit UI components
│   ├── __init__.py
│   └── dashboard.py        # All tab rendering functions
│
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

```
Upload CSV
    │
    ▼
Parse Goal (keyword + semantic)
    │
    ▼
Profile Dataset (column stats, null %, unique counts)
    │
    ▼
Infer Column Roles (identifier / date / target / feature / …)
    │
    ▼
Detect Target Column & Task Type (classification or regression)
    │
    ▼
Data Readiness Report (score, leakage risk, missing signals)
    │
    ▼
Train Selected Models (up to 4, with proper preprocessing)
    │
    ▼
Evaluate & Compare Models (metrics table + bar chart)
    │
    ▼
Generate Feature Importance (SHAP → Plotly → Seaborn)
    │
    ▼
Final Plain-English Summary (downloadable)
```

---

## 🗂 Example Datasets to Try

| Dataset | Goal to type |
|---|---|
| [Titanic](https://www.kaggle.com/c/titanic/data) | `predict whether a passenger survived` |
| [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | `predict house sale price` |
| [Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | `predict customer churn` |
| [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `detect fraudulent transactions` |

---

## ⚙️ Configuration

| Setting | Default | Where to change |
|---|---|---|
| Max upload size | 500 MB | `dia/utils.py → MAX_FILE_BYTES` |
| Max models | 4 | `dia/utils.py → MAX_MODELS` |
| Test split ratio | 20% | `dia/model_trainer.py → test_size` |
| SHAP sample size | 200 rows | `dia/explainability.py → _try_shap()` |
| Top features shown | 20 | `app.py → generate_explanation(top_n=20)` |

---

## 🔒 Privacy

- **No data is stored**. Uploaded CSV files are read into memory and discarded when the session ends.
- No network calls are made with your data.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| UI / Server | [Streamlit](https://streamlit.io) |
| Data Handling | [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Machine Learning | [scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.ai) *(opt)*, [LightGBM](https://lightgbm.readthedocs.io) *(opt)* |
| Explainability | [SHAP](https://shap.readthedocs.io) *(opt)* |
| Visualisation | [Plotly](https://plotly.com/python/), [Seaborn](https://seaborn.pydata.org) / Matplotlib |
| Semantic NLP | [sentence-transformers](https://www.sbert.net) *(opt)* |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — feel free to use and adapt this project.
