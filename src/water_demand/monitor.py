"""Stage 4, part 1: has the world changed since v1 was trained?

- PSI for every numeric column: cleaned train.csv vs later.csv
- v1's MAE on later.csv, separately for each tower
Read-only: this file changes nothing in MLflow or models/.
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from src.water_demand.data import clean, load_raw


def psi(expected, actual, bins=10):
    """Population Stability Index between two samples of one column.

    Bin edges come from the TRAINING data's quantiles (10 bins), then we
    compare what share of rows falls in each bin in train vs later.
    PSI = sum((later% - train%) * ln(later% / train%)).
    Rough guide: < 0.1 stable, 0.1-0.25 some shift, > 0.25 big shift.
    """
    expected = pd.Series(expected).dropna().to_numpy()
    actual = pd.Series(actual).dropna().to_numpy()

    # Inner edges at the 10%, 20%, ... 90% quantiles of train.
    # np.unique: whole-number columns can give the same edge twice.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e_counts = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=len(edges) + 1)
    a_counts = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=len(edges) + 1)

    eps = 1e-6  # avoids log(0) when a bin is empty
    e_pct = np.clip(e_counts / e_counts.sum(), eps, None)
    a_pct = np.clip(a_counts / a_counts.sum(), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def main():
    params = yaml.safe_load(open("params.yaml"))
    numeric = params["features"]["numeric"]
    features = numeric + params["features"]["categorical"]
    target = params["target"]

    train = clean(load_raw(params["data"]["train"]), verbose=False)
    later = load_raw(params["data"]["later"])  # later.csv is already clean

    # 1. Drift: PSI per numeric column.
    print("PSI (cleaned train vs later, 10 bins)")
    scores = {col: psi(train[col], later[col]) for col in numeric}
    for col, value in scores.items():
        print(f"  {col:24s} {value:.4f}")
    print(f"Most drifted feature: {max(scores, key=scores.get)}\n")

    # 2. v1's error on later.csv, per tower. Load v1 from the registry.
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    name = params["mlflow"]["registered_model_name"]
    v1 = mlflow.sklearn.load_model(f"models:/{name}/1")

    later = later.assign(pred=v1.predict(later[features]))
    print("v1 MAE on later.csv by tower")
    by_group = {
        tower: mean_absolute_error(g[target], g["pred"])
        for tower, g in later.groupby("tower")
    }
    for tower, value in by_group.items():
        print(f"  {tower:8s} {value:,.2f}")
    print(f"Worst served group: {max(by_group, key=by_group.get)}")


if __name__ == "__main__":
    main()
