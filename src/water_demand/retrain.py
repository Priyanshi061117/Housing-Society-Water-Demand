"""Stage 4, part 2: retrain v2, apply the client's promotion rule,
and predict the holdout file with whichever model is champion.

Run this ONCE: it registers version 2 in the MLflow registry.
"""
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.water_demand.data import clean, load_raw, sha256_of
from src.water_demand.train import build_pipeline

PROMOTE_IF_BETTER_BY = 0.12  # client rule: v2 must be at least 12% better


def main():
    params = yaml.safe_load(open("params.yaml"))
    features = params["features"]["numeric"] + params["features"]["categorical"]
    target = params["target"]
    name = params["mlflow"]["registered_model_name"]

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    client = MlflowClient()

    # Safety: stop if v2 already exists, so re-running can't make version 3.
    if len(client.search_model_versions(f"name='{name}'")) >= 2:
        raise SystemExit("Version 2 already registered - not retraining again.")

    # v1 = version 1 in the registry. Reuse its winning settings for v2,
    # so the only thing that changes is the training data.
    v1_run_id = client.get_model_version(name, "1").run_id
    v1_name = client.get_run(v1_run_id).info.run_name
    cfg = next(m for m in params["models"] if m["name"] == v1_name)
    v1 = mlflow.sklearn.load_model(f"models:/{name}/1")

    # Split later.csv BY DATE: first 70% to learn from, last 30% to check.
    train = clean(load_raw(params["data"]["train"]), verbose=False)
    later = load_raw(params["data"]["later"]).sort_values("date", kind="stable").reset_index(drop=True)
    cut = int(len(later) * 0.7)
    later_fit, check = later.iloc[:cut], later.iloc[cut:]
    print(f"later.csv: {len(later_fit)} rows to retrain on "
          f"({later_fit.date.min()} to {later_fit.date.max()}), "
          f"{len(check)} rows to check ({check.date.min()} to {check.date.max()})")

    # v2: same settings as v1, trained on cleaned train + first 70% of later.
    fit_data = pd.concat([train[features + [target]], later_fit[features + [target]]])
    with mlflow.start_run(run_name=f"v2_{cfg['name']}") as run:
        v2 = build_pipeline(params["features"]["numeric"], params["features"]["categorical"],
                            cfg["type"], cfg["params"], params["split"]["random_state"])
        v2.fit(fit_data[features], fit_data[target])

        mae_v1 = mean_absolute_error(check[target], v1.predict(check[features]))
        mae_v2 = mean_absolute_error(check[target], v2.predict(check[features]))

        mlflow.log_param("data_sha256", sha256_of(params["data"]["train"]))
        mlflow.log_param("model_type", cfg["type"])
        mlflow.log_params(cfg["params"])
        mlflow.log_param("trained_on", "train.csv + first 70% of later.csv")
        mlflow.log_metric("mae_v1_on_check", mae_v1)
        mlflow.log_metric("mae_v2_on_check", mae_v2)
        mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle")
        v2_run_id = run.info.run_id

    # Register v2 either way (the tool wants 2+ versions); only the alias moves.
    version = mlflow.register_model(f"runs:/{v2_run_id}/model", name).version

    improvement = (mae_v1 - mae_v2) / mae_v1
    decision = "promote" if improvement >= PROMOTE_IF_BETTER_BY else "keep"
    print(f"\nMAE on last 30%:  v1 = {mae_v1:,.2f}   v2 = {mae_v2:,.2f}")
    print(f"v2 is {improvement:.1%} better (rule: at least {PROMOTE_IF_BETTER_BY:.0%})")
    print(f"Decision: {decision}")

    if decision == "promote":
        client.set_registered_model_alias(name, "champion", version)
        joblib.dump(v2, params["model_path"])  # the API now serves v2
        print(f"Champion is now version {version}; {params['model_path']} updated")
    else:
        print("Champion stays version 1")

    # Final predictions with whoever is champion now.
    champion = mlflow.sklearn.load_model(f"models:/{name}@champion")
    holdout = load_raw(params["data"]["holdout"])
    out = pd.DataFrame({"record_id": holdout["record_id"],
                        "prediction": champion.predict(holdout[features])})
    Path("predictions").mkdir(exist_ok=True)
    out.to_csv("predictions/holdout_predictions.csv", index=False)
    print(f"Saved predictions/holdout_predictions.csv ({len(out)} rows)")
    print(f"v2 run id: {v2_run_id}, registered as version {version}")


if __name__ == "__main__":
    main()
