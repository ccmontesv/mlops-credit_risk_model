from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution
import shap
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os
import warnings
import base64, pickle

warnings.filterwarnings("ignore")



# DAG setup
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 6, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='credit_risk_ml_pipeline',
    default_args=default_args,
    schedule_interval='@monthly',
    catchup=False,
    tags=['ml', 'credit', 'xgboost'],
)



# --- Preprocessing Function ---
def preprocess_data(**kwargs):

    PATH = '/opt/airflow/data/credit_risk_dataset.csv'
    df = pd.read_csv(PATH)
    df.rename(columns={"loan_status": "target"}, inplace=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    string_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cols_to_process = [col for col in numeric_cols if col != "target"]

    for col in cols_to_process:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        skew_val = df[col].dropna().skew()
        impute_val = df[col].median() if abs(skew_val) > 0.5 else df[col].mean()
        df[col] = df[col].fillna(impute_val)
        if col == 'person_age':
            df[col] = df[col].clip(lower=18, upper=90)
        elif col == 'loan_percent_income':
            df[col] = df[col].clip(upper=1.0).astype(float)
        else:
            upper_bound = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=0, upper=upper_bound).astype(float)

    label_cols = ['loan_grade', 'cb_person_default_on_file']
    for col in label_cols:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
    df['loan_grade'] = OrdinalEncoder(categories=[['A', 'B', 'C', 'D', 'E', 'F', 'G']]).fit_transform(df[['loan_grade']])
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

    onehot_cols = ['person_home_ownership', 'loan_intent']
    for col in onehot_cols:
        df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
        df[col] = df[col].fillna("Unknown")
        rare_labels = df[col].value_counts(normalize=True)[lambda x: x < 0.01].index
        df[col] = df[col].replace(rare_labels, 'Other')

    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype=int)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    ti = kwargs['ti']
    ti.xcom_push(key='X_train', value=base64.b64encode(pickle.dumps(X_train)).decode())
    ti.xcom_push(key='X_test', value=base64.b64encode(pickle.dumps(X_test)).decode())
    ti.xcom_push(key='y_train', value=base64.b64encode(pickle.dumps(y_train)).decode())
    ti.xcom_push(key='y_test', value=base64.b64encode(pickle.dumps(y_test)).decode())
    ti.xcom_push(key='scale_pos_weight', value=scale_pos_weight)


# --- Training with Optuna ---
def train_model(**kwargs):

    ti = kwargs['ti']
    decode = lambda k: pickle.loads(base64.b64decode(ti.xcom_pull(key=k, task_ids='preprocess_data')))
    X_train, y_train = decode('X_train'), decode('y_train')
    scale_pos_weight = ti.xcom_pull(key='scale_pos_weight', task_ids='preprocess_data')

    pipeline = Pipeline([
        ("classifier", XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight, tree_method="hist"))
    ])
    param_space = {
        "classifier__n_estimators": IntDistribution(50, 150, step=25),
        "classifier__max_depth": IntDistribution(3, 10),
        "classifier__learning_rate": FloatDistribution(0.01, 0.3),
        "classifier__subsample": FloatDistribution(0.5, 1.0),
        "classifier__colsample_bytree": FloatDistribution(0.5, 1.0),
    }
    optuna_search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=param_space,
        cv=StratifiedKFold(5),
        n_trials=20,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )
    optuna_search.fit(X_train, y_train)
    #best_model = optuna_search.best_estimator_
    #best_params = optuna_search.best_params_

    ti.xcom_push(key='best_model', value=base64.b64encode(pickle.dumps(optuna_search.best_estimator_)).decode())
    ti.xcom_push(key='best_params', value=optuna_search.best_params_)



# --- Feature Importance + Retraining ---
def feature_selection(**kwargs):


    ti = kwargs['ti']
    decode = lambda k: pickle.loads(base64.b64decode(ti.xcom_pull(key=k, task_ids='preprocess_data')))
    decode_train = lambda k: pickle.loads(base64.b64decode(ti.xcom_pull(key=k, task_ids='train_model')))
    X_train, X_test, y_train, y_test = map(decode, ['X_train', 'X_test', 'y_train', 'y_test'])
    best_model = decode_train('best_model')
    best_params = ti.xcom_pull(key='best_params', task_ids='train_model')
    scale_pos_weight = ti.xcom_pull(key='scale_pos_weight', task_ids='preprocess_data')

    drop_threshold = 0.01
    min_features = 3
    re_shap = True
    baseline_model = best_model.named_steps["classifier"]
    baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])
    explainer = shap.TreeExplainer(baseline_model)
    shap_values = explainer.shap_values(X_test)
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    features = feature_importance["feature"].tolist()
    results = []

    while len(features) >= min_features:
        X_train_sub = X_train[features]
        X_test_sub = X_test[features]
        model_sub = XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight, tree_method="hist",
                                  **{k.split('__')[1]: v for k, v in best_params.items()})
        model_sub.fit(X_train_sub, y_train)
        auc = roc_auc_score(y_test, model_sub.predict_proba(X_test_sub)[:, 1])
        results.append({"n_features": len(features), "features": features.copy(), "auc": auc})
        if baseline_auc - auc > drop_threshold:
            break
        if re_shap:
            explainer = shap.TreeExplainer(model_sub)
            shap_values = explainer.shap_values(X_test_sub)
            feature_importance = pd.DataFrame({
                "feature": X_test_sub.columns,
                "importance": np.abs(shap_values).mean(axis=0)
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
            features = feature_importance["feature"].tolist()[:-1]
        else:
            features = features[:-1]

    df_feature_importance = pd.DataFrame(results)
    best_row = df_feature_importance.loc[df_feature_importance["auc"].idxmax()]
    best_features = best_row["features"]
    X_train_top = X_train[best_features]
    X_test_top = X_test[best_features]
    final_model = XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight, tree_method="hist",
                                **{k.split('__')[1]: v for k, v in best_params.items()})
    final_model.fit(X_train_top, y_train)
    y_pred_top = final_model.predict(X_test_top)
    y_proba_top = final_model.predict_proba(X_test_top)[:, 1]

    # Save results via XCom
    ti.xcom_push(key='best_features', value=base64.b64encode(pickle.dumps(best_features)).decode())
    ti.xcom_push(key='final_model', value=base64.b64encode(pickle.dumps(final_model)).decode())

    print("Accuracy:", accuracy_score(y_test, y_pred_top))
    print("Precision:", precision_score(y_test, y_pred_top))
    print("Recall:", recall_score(y_test, y_pred_top))
    print("F1 Score:", f1_score(y_test, y_pred_top))
    print("ROC AUC:", roc_auc_score(y_test, y_proba_top))

# --- Tracking with MLflow ---
def track_model(**kwargs):

    ti = kwargs['ti']
    decode = lambda key, task: pickle.loads(base64.b64decode(ti.xcom_pull(key=key, task_ids=task)))

    X_test = decode('X_test', 'preprocess_data')
    y_test = decode('y_test', 'preprocess_data')
    best_features = decode('best_features', 'feature_selection')
    final_model = decode('final_model', 'feature_selection')
    scale_pos_weight = ti.xcom_pull(key='scale_pos_weight', task_ids='preprocess_data')
    best_params = ti.xcom_pull(key='best_params', task_ids='train_model')

    X_test_top = X_test[best_features]
    y_pred_top = final_model.predict(X_test_top)
    y_proba_top = final_model.predict_proba(X_test_top)[:, 1]

    model_name = "CreditRisk_XGB"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_version_name = f"{model_name}_{timestamp}"

    mlflow.set_tracking_uri("file:///opt/airflow/mlruns")
    mlflow.set_experiment("CreditRiskTracking")

    with mlflow.start_run(run_name=model_version_name) as run:

        signature = infer_signature(X_test_top, y_pred_top)
        mlflow.sklearn.log_model(final_model, artifact_path="model", signature=signature, registered_model_name=model_name)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred_top),
            "precision": precision_score(y_test, y_pred_top),
            "recall": recall_score(y_test, y_pred_top),
            "f1_score": f1_score(y_test, y_pred_top),
            "roc_auc": roc_auc_score(y_test, y_proba_top)
        })
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("best_n_features", len(best_features))
        for param, val in best_params.items():
            mlflow.log_param(param, val)

    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{model_name}'")
    new_auc = roc_auc_score(y_test, y_proba_top)
    prod_version = None
    best_score = -1
    for version in all_versions:
        details = client.get_model_version(model_name, version.version)
        if details.tags.get("production") == "true":
            prod_version = details
            best_score = client.get_run(details.run_id).data.metrics.get("roc_auc", -1)
            break
    new_version = max([int(v.version) for v in all_versions])
    if prod_version is None:
        client.set_model_version_tag(model_name, new_version, "production", "true")
    elif new_auc > best_score:
        client.delete_model_version_tag(model_name, prod_version.version, "production")
        client.set_model_version_tag(model_name, new_version, "production", "true")

# Define Airflow tasks
preprocess_task = PythonOperator(
    task_id='preprocess_data', 
    python_callable=preprocess_data, 
    provide_context=True,
    dag=dag
    )
train_task = PythonOperator(
    task_id='train_model', 
    python_callable=train_model, 
    provide_context=True, 
    dag=dag
    )
feature_task = PythonOperator(
    task_id='feature_selection', 
    python_callable=feature_selection, 
    provide_context=True, 
    dag=dag
    )
tracking_task = PythonOperator(
    task_id='track_model', 
    python_callable=track_model, 
    provide_context=True,
    dag=dag
    )


# Set task dependencies
preprocess_task >> train_task >> feature_task >> tracking_task

#http://localhost:8080  <- to open the airflow ui
