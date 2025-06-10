# Credit Risk ML Pipeline

This project implements a **machine learning pipeline** to assess credit risk using modern MLOps practices. The pipeline is orchestrated with **Apache Airflow**, packaged with **Docker**, and includes **model optimization** using **Optuna**, **feature importance analysis** using **SHAP**, and **model tracking** via **MLflow**.

## 🧩 Project Structure

```
.
├── dags/
│   └── credit_risk_ml_pipeline.py   # Main DAG definition
├── data/
│   └── credit_risk_dataset.csv      # Input dataset
├── mlruns/                          # MLflow tracking data (if mapped)
├── Dockerfile                       # (Optional) Custom Docker setup
├── docker-compose.yaml              # Multi-service container configuration
└── README.md                        # Project overview
```

## ⚙️ Components

### 🔁 Airflow
- Orchestrates the ML steps:
  - Preprocessing
  - Hyperparameter tuning with Optuna
  - SHAP-based feature selection
  - Model retraining and evaluation
  - MLflow tracking

### 📦 Docker
- Provides a reproducible environment with all required dependencies (e.g., scikit-learn, XGBoost, MLflow, Optuna).

### 🔍 Optuna
- Performs hyperparameter optimization using `OptunaSearchCV`.

### 📊 MLflow
- Tracks metrics, parameters, and model versions.
- Automatically compares AUC and promotes best models to "production" stage.

### 📈 SHAP
- Computes feature importances to guide model retraining on top-ranked features.

## 🚀 Running the Pipeline

1. **Start Docker & Airflow**:
   ```bash
   docker-compose up
   ```

2. **Access Airflow UI**:
   - Visit: `http://localhost:8080`
   - Default login:
     - **Username**: `airflow`
     - **Password**: `airflow`

3. **Trigger DAG**:
   - Go to Airflow UI → Enable & Trigger `credit_risk_ml_pipeline`

4. **Access MLflow UI** (if `mlruns` is mounted externally):
   ```bash
   mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
   ```
   Visit: `http://localhost:5000`

## 📝 Dataset

- Source: `data/credit_risk_dataset.csv`
- Target variable: `loan_status` (renamed internally to `target`)
- Includes categorical, numerical, and skewed variables cleaned before training.

## 🧠 Model

- Trained using `XGBClassifier` with optimized parameters.
- Evaluated on metrics like Accuracy, F1, ROC AUC, etc.
- Best-performing models automatically registered and versioned with MLflow.

## 📌 Requirements

If running outside Docker:

- Python 3.10+
- `mlflow`, `xgboost`, `optuna`, `scikit-learn`, `shap`, `matplotlib`, `pandas`, `airflow`

> Recommended: run using Docker Compose for a fully self-contained environment.

---

## 🙋‍♂️ Author

Created as part of a modular ML pipeline demonstration using MLOps tools.

Feel free to fork and adapt it for your own binary classification problems.