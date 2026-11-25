"""Train v1: try every model setting in params.yaml, track each run in
MLflow, register the best one and give it the 'champion' alias."""
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.water_demand.data import clean, load_raw, sha256_of

# Map the 'type' names used in params.yaml to scikit-learn classes.
REGRESSORS = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def build_pipeline(numeric, categorical, model_type, model_params, random_state):
    """Preprocessing + regressor in one object, so the API can pass raw
    request fields straight to .predict()."""
    preprocess = ColumnTransformer([
        # Fill missing numbers (e.g. the old -999 sentinels) with the median.
        ("num", SimpleImputer(strategy="median"), numeric),
        # Turn text categories into 0/1 columns; ignore unseen values safely.
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = REGRESSORS[model_type](random_state=random_state, **model_params)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main():
    params = yaml.safe_load(open("params.yaml"))
    numeric = params["features"]["numeric"]
    categorical = params["features"]["categorical"]
    features = numeric + categorical
    target = params["target"]
    split = params["split"]

    # Hash of the ORIGINAL file, so every run proves which data it used.
    data_sha256 = sha256_of(params["data"]["train"])

    df = clean(load_raw(params["data"]["train"]))
    X_train, X_val, y_train, y_val = train_test_split(
        df[features], df[target],
        test_size=split["test_size"], random_state=split["random_state"],
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    results = []  # (mae, run_id, fitted pipeline, name)
    for cfg in params["models"]:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            pipe = build_pipeline(numeric, categorical, cfg["type"],
                                  cfg["params"], split["random_state"])
            pipe.fit(X_train, y_train)
            mae = mean_absolute_error(y_val, pipe.predict(X_val))

            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", cfg["type"])
            mlflow.log_params(cfg["params"])
            mlflow.log_metric("mae_val", mae)
            mlflow.sklearn.log_model(pipe, name="model",
                                     serialization_format="cloudpickle")

            print(f"{cfg['name']:10s}  mae_val = {mae:,.0f} litres  run_id = {run.info.run_id}")
            results.append((mae, run.info.run_id, pipe, cfg["name"]))

    # Pick the lowest MAE and register it ONCE -> version 1 = v1.
    best_mae, best_run_id, best_pipe, best_name = min(results, key=lambda r: r[0])
    name = params["mlflow"]["registered_model_name"]
    version = mlflow.register_model(f"runs:/{best_run_id}/model", name).version
    MlflowClient().set_registered_model_alias(name, "champion", version)

    # The same fitted pipeline, saved where the API will load it.
    Path(params["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, params["model_path"])

    print(f"\nBest: {best_name}  mae_val = {best_mae:,.0f}  run_id = {best_run_id}")
    print(f"Registered '{name}' version {version} with alias 'champion'")
    print(f"Saved {params['model_path']}")


if __name__ == "__main__":
    main()
