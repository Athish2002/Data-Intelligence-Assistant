# 🧠 Data Intelligence Assistant

![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-informational)

> [!WARNING]
> **This project is a work in progress.** It is under active development and
> not production-ready. APIs, module layout, and the UI change frequently,
> features may be incomplete or broken between commits, and parts of this
> README lag behind the code. Use it to explore and experiment, not to rely on.

> An intelligent data science assistant that helps you upload a CSV dataset, describe your machine-learning goal in plain English, and receive a full diagnostic report plus baseline model training — all inside a clean, modern web interface.

---

## 🚧 Project Status

Actively being built. Rough edges to expect right now:

- **Docs drift** — the Features table below still describes an earlier, smaller version
  of the app (e.g. "choose up to 4 models"); the trainer now supports 10 model families,
  plus deep, causal, streaming, time-series, and bandit engines that aren't documented yet.
- **Undeclared deps** — `torch` (deep autoencoder) and `catboost` are imported by some
  modules, and the Fullstack Edition needs `fastapi` / `uvicorn` / `pydantic` /
  `python-multipart`. None of these are in `requirements.txt` yet; install them by hand.
- **Unstable interfaces** — function signatures in `dia/` are not frozen; treat nothing
  as a stable public API yet.
- **Partial coverage** — tests exist but do not cover every module.

Issues and PRs are welcome, but expect the ground to move under you.

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

## 🖥️ Two Editions

The same `dia/` engine powers two independent front ends. Pick whichever you prefer —
both read the same code and produce the same results.

| | **Streamlit Edition** | **Fullstack Edition** |
|---|---|---|
| Launch | `python run_streamlit.py` | `python run_fullstack.py` |
| URL | `http://localhost:8501` | `http://localhost:8000` |
| Stack | Streamlit (Python-rendered UI) | FastAPI backend + vanilla-JS client |
| Entry point | `app.py` + `ui/` | `api/server.py` + `frontend/` |
| API docs | — | Swagger UI at `/docs` |
| Best for | Fast local exploration, notebooks-style use | Integrating over REST, custom/embedded UI |

Extra dependencies for the Fullstack Edition (not yet in `requirements.txt` — see
[Project Status](#-project-status)):

```bash
pip install fastapi uvicorn pydantic python-multipart
```

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

Streamlit Edition:
```bash
python run_streamlit.py
```

Fullstack Edition (FastAPI + JS client):
```bash
python run_fullstack.py
```

The Streamlit app opens at `http://localhost:8501`; the fullstack app serves the web
client at `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`.

---

## 📁 Project Structure

```
data-intelligence-assistant/
│
├── run_streamlit.py        # Launcher — Streamlit Edition (:8501)
├── run_fullstack.py        # Launcher — FastAPI + JS Edition (:8000)
│
├── app.py                  # Streamlit entry point
├── ui/                     # Streamlit UI components
│   └── dashboard.py        # Tab rendering functions
│
├── api/                    # REST backend (Fullstack Edition)
│   ├── server.py           # FastAPI app, routes, static file serving
│   └── schemas.py          # Pydantic request/response models
│
├── frontend/               # Web client (Fullstack Edition)
│   ├── index.html
│   ├── css/style.css       # Glassmorphic styling
│   └── js/{app.js,api.js}  # UI logic and REST client
│
├── dia/                    # Shared backend engine (used by BOTH editions)
│   ├── data_loader.py      # Dataset loading and size checks
│   ├── ingestion/          # CSV, SQL, S3, GCS, Azure, BigQuery, Snowflake, URL
│   ├── goal_parser.py      # Keyword + semantic goal parsing
│   ├── data_profiler.py    # Column profiling, role inference, readiness report
│   ├── data_quality.py     # Validation, expectations, sanitization
│   ├── feature_engineer.py # AutoFE and mutual-information selection
│   ├── model_trainer.py    # 10 model families, HPO, calibration, ensembling
│   ├── deep_autoencoder.py # PyTorch tabular autoencoder (anomaly detection)
│   ├── time_series.py      # Lag/rolling features + autoregressive forecasting
│   ├── streaming_learner.py# Online learning via partial_fit
│   ├── bandit_optimizer.py # LinUCB / Thompson Sampling
│   ├── causal_engine.py    # Counterfactuals and T-learner uplift
│   ├── explainability.py   # SHAP / Plotly / Seaborn importance charts
│   ├── fairness.py         # Bias auditing and rule extraction
│   ├── drift_monitor.py    # Distribution drift and IsolationForest outliers
│   ├── graph_engine.py     # NetworkX relational analytics
│   ├── mlops_registry.py   # Model registry and versioning
│   └── ...                 # See dia/ for the full module list
│
├── tests/
├── k8s/                    # Deployment manifests
├── requirements.txt
└── README.md
```

> The tree above is abridged — `dia/` contains roughly 40 modules. Browse
> [`dia/`](dia/) for the authoritative list.

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
| UI / Server | [Streamlit](https://streamlit.io) *(Streamlit Edition)* |
| REST API | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) *(Fullstack Edition)* |
| Web Client | Vanilla HTML / CSS / JavaScript *(Fullstack Edition)* |
| Data Handling | [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Machine Learning | [scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.ai) *(opt)*, [LightGBM](https://lightgbm.readthedocs.io) *(opt)*, [CatBoost](https://catboost.ai) *(opt)* |
| Deep Learning | [PyTorch](https://pytorch.org) *(opt — tabular autoencoder)* |
| Graph Analytics | [NetworkX](https://networkx.org) |
| Explainability | [SHAP](https://shap.readthedocs.io) *(opt)* |
| Visualisation | [Plotly](https://plotly.com/python/), [Seaborn](https://seaborn.pydata.org) / Matplotlib |
| Semantic NLP | [sentence-transformers](https://www.sbert.net) *(opt)* |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — feel free to use and adapt this project.
