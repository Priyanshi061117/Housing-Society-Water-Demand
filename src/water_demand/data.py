"""Loading and cleaning the raw water-demand data.

The raw CSV files are never changed on disk. Every fix happens in memory,
so the original file (and its hash) stays exactly as it came in the zip.
"""
import hashlib

import numpy as np
import pandas as pd

# Values found by our data investigation (reports/findings.yaml).
LEAK_COLUMN = "meter_litres_next_morning"
SENTINEL = -999
SQFT_PER_SQM = 10.764
UNIT_MIXUP_TOWER = "Tower A"


def sha256_of(path):
    """SHA-256 of the file bytes. Logged in every MLflow run as data_sha256."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_raw(path):
    """Read a CSV exactly as it is."""
    return pd.read_csv(path)


def clean(df, verbose=True):
    """Return a cleaned copy of train.csv. The order of the steps matters."""
    df = df.copy()
    start = len(df)

    # 1. Duplicate rows: the same record typed twice. Keep one copy.
    df = df.drop_duplicates()
    dups = start - len(df)

    # 2. Target typos: a few targets were typed with an extra zero.
    #    The next-morning meter reading is about the same as the true target,
    #    so a target/meter ratio of about 10 shows the typo. Divide by 10.
    ratio = df["litres_per_day"] / df[LEAK_COLUMN]
    typo = ratio.between(9.5, 10.5)
    df.loc[typo, "litres_per_day"] = df.loc[typo, "litres_per_day"] / 10

    # 3. Unit mix-up: Tower A wrote garden area in square feet, not m².
    #    Convert ALL Tower A rows (every one of them is a multiple of 10.764).
    tower_a = df["tower"] == UNIT_MIXUP_TOWER
    df.loc[tower_a, "garden_area_m2"] = df.loc[tower_a, "garden_area_m2"] / SQFT_PER_SQM

    # 4. Sentinel: -999 means "no reading". Make it a real missing value.
    #    We do NOT fill it here; the model pipeline's imputer does that,
    #    so the API treats missing values the same way.
    sentinel = df["supply_hours"] == SENTINEL
    df.loc[sentinel, "supply_hours"] = np.nan

    # 5. Leaking column: only known the next morning, so it cannot be a
    #    feature. Dropped last because step 2 needs it.
    df = df.drop(columns=[LEAK_COLUMN])

    if verbose:
        print(f"Rows in                 : {start}")
        print(f"Duplicates removed      : {dups}")
        print(f"Target typos fixed      : {int(typo.sum())}")
        print(f"Tower A rows converted  : {int(tower_a.sum())}")
        print(f"Sentinels set to NaN    : {int(sentinel.sum())}")
        print(f"Dropped leaking column  : {LEAK_COLUMN}")
        print(f"Rows out                : {len(df)}")
    return df


if __name__ == "__main__":
    # Quick check: python -m src.water_demand.data
    import yaml

    params = yaml.safe_load(open("params.yaml"))
    cleaned = clean(load_raw(params["data"]["train"]))
    print(cleaned.describe().round(1).T)
