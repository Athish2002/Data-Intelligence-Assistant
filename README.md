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

- **Uneven maturity** — the Features list is accurate, but depth varies a lot between
  entries. Core profiling and AutoML are the most exercised paths; several governance
  and adaptive engines are closer to working prototypes than finished features.
- **Optional deps degrade silently** — `torch` (Deep Autoencoder), `catboost`, `shap`,
  and `sentence-transformers` are commented out in `requirements.txt`. Those features
  report that the dependency is missing rather than failing outright.
- **Unstable interfaces** — function signatures in `dia/` are not frozen; treat nothing
  as a stable public API yet.
- **Partial coverage** — tests exist but do not cover every module.

Issues and PRs are welcome, but expect the ground to move under you.

---

## ✨ Features

Organised roughly the way the app presents them.

### 📊 Core Intelligence
| Feature | Description |
|---|---|
| **Smart Dataset Profiling** | Detects column types, null percentages, unique counts, high-cardinality columns, and likely identifier / date columns |
| **Column Role Inference** | Labels each column as identifier, date, duration, target candidate, numeric feature, etc. — with a confidence score |
| **Dual Goal Parsing** | Maps free-text goals to a task type via **keyword matching** plus optional **semantic matching** (sentence-transformers) |
| **Automatic Task Detection** | Infers classification vs. regression from the goal text and the target's dtype and unique-value ratio |
| **Data Readiness Audit** | Scores the dataset 0–100, highlights useful features, flags leakage risks, lists missing signals |
| **Data Sanitization & Quality** | Cleaning, validation rules, and expectation/contract checks |
| **Multi-Source Ingestion** | Local CSV, URL, SQL, S3, GCS, Azure Blob, BigQuery, Snowflake |

### 🤖 AutoML & Explainability
| Feature | Description |
|---|---|
| **10 Model Families** | Logistic/Linear (Ridge), Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM, KNN, and MLP — each with classification and regression variants |
| **Hyperparameter Optimization** | `RandomizedSearchCV` over per-model search grids, with cross-validation |
| **Calibration & Ensembling** | Sigmoid probability calibration and a soft-voting ensemble of the top 3 models |
| **Automated Feature Engineering** | Generated features plus mutual-information based selection |
| **Class Imbalance Handling** | Sample weighting computed from class frequencies |
| **Model Leaderboard** | Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision · MAE, RMSE, R² |
| **Feature Explainability** | SHAP (Tree/Linear/Kernel explainers) with graceful fallback to native importances |
| **Business ROI & Impact** | Translates confusion-matrix outcomes into cost/benefit terms |
| **Interactive Simulator** | Change feature values and watch predictions update live |
| **Active Learning Queue** | Surfaces highest-uncertainty predictions for human review |
| **A/B Test Planner** | Experiment sizing and significance testing |

### 🧪 Advanced & Adaptive Engines
| Feature | Description |
|---|---|
| **Deep Tabular Autoencoder** | PyTorch bottleneck network; flags anomalies by reconstruction error with per-feature attribution *(optional — needs `torch`)* |
| **Causal ML** | Counterfactual "what-if" search and T-learner uplift modelling |
| **Contextual Bandits** | LinUCB and Thompson Sampling for exploration/exploitation policy simulation |
| **Time-Series Forecasting** | Auto-detects date columns, builds lag/rolling/calendar features, forecasts with prediction intervals |
| **Streaming / Online Learning** | Incremental `partial_fit` learners (SGD, Passive-Aggressive) with live accuracy curves |
| **NLP & Lexical Analysis** | TF-IDF vectorization plus TruncatedSVD topic extraction |
| **Graph Intelligence** | NetworkX entity graphs — PageRank, betweenness, community and collusion-ring detection |
| **Synthetic Data & Privacy** | Gaussian-copula synthesis with calibrated Laplace noise for differential privacy |

### 🛡️ Governance, MLOps & Production
| Feature | Description |
|---|---|
| **Fairness & Bias Auditing** | Group metrics plus decision-tree rule extraction to explain disparities |
| **Drift Monitoring** | Distribution drift detection and `IsolationForest` outlier flagging |
| **Model Registry** | Versioning and promotion tracking, with canary routing |
| **Feature Store** | Feature definitions and retrieval |
| **Privacy & GDPR Audit** | PII detection and compliance reporting |
| **RBAC** | Role-based access control primitives |
| **SQL Transpiler** | Compiles trained decision trees into in-database `CASE` statements |
| **Production Code Export** | Generates a runnable training script matching your chosen configuration |
| **Executive Briefing** | Plain-English summary of what was inferred, which model won, and whether the data is trustworthy — downloadable as Markdown |
| **AI Chat Copilot** | Natural-language queries over the dataset; deterministic offline mode by default, optional LLM reasoning |

> ⚠️ Not every feature above is equally mature — see [Project Status](#-project-status).

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

Both editions install from the same `requirements.txt`; the FastAPI stack is
included, so no extra step is needed.

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

> **Optional extras** (commented out in `requirements.txt` — uncomment to enable):
> - `torch` — Deep Tabular Autoencoder (large download)
> - `shap` — SHAP-based explainability
> - `sentence-transformers` — semantic goal parsing (also pulls in torch)
> - `catboost` — CatBoost model option
>
> Each is loaded lazily: without it, the matching feature reports the missing
> dependency instead of breaking the app.

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
