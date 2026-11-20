
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

train = pd.read_csv(DATA / "train.csv")
later = pd.read_csv(DATA / "later.csv")

print("=" * 60)
print("G017 MP071 - DATA INVESTIGATION")
print("=" * 60)

# --------------------------------------------------
# 1. POTENTIAL DATA LEAKAGE
# --------------------------------------------------
print("\n[1] POTENTIAL DATA LEAKAGE")

target = "litres_per_day"
candidate = "meter_litres_next_morning"

if candidate in train.columns:
    valid = train[[target, candidate]].dropna()

    print("Correlation:", valid[target].corr(valid[candidate]))

    ratio = valid[target] / valid[candidate]
    print("Median target/candidate ratio:", ratio.median())

    within_5_percent = (
        (valid[target] - valid[candidate]).abs()
        / valid[target].abs()
    ) <= 0.05

    print("Records within 5%:", within_5_percent.sum())
    print("Total valid records:", len(valid))
    print("Later dataset columns:", list(later.columns))

# --------------------------------------------------
# 2. UNIT MIX-UP
# --------------------------------------------------
print("\n[2] GARDEN AREA UNIT INVESTIGATION")

if "tower" in train.columns:
    print("\nGarden area statistics by tower:")
    print(train.groupby("tower")["garden_area_m2"].describe())

    print("\nGarden area medians:")
    print(train.groupby("tower")["garden_area_m2"].median())

# --------------------------------------------------
# 3. SENTINEL VALUES
# --------------------------------------------------
print("\n[3] SENTINEL VALUE INVESTIGATION")

for col in train.columns:
    count = (train[col] == -999).sum()

    if count > 0:
        print(f"{col}: {count} occurrences of -999")

# --------------------------------------------------
# 4. DUPLICATE ROWS
# --------------------------------------------------
print("\n[4] DUPLICATE ROW INVESTIGATION")

print("Exact duplicate rows:", train.duplicated().sum())

# --------------------------------------------------
# 5. STUCK SENSOR
# --------------------------------------------------
print("\n[5] STUCK SENSOR INVESTIGATION")

numeric_cols = train.select_dtypes(include="number").columns

for col in numeric_cols:
    if col == "record_id":
        continue

    values = train[col].dropna()

    if values.nunique() <= 10:
        continue

    # Consecutive identical-value runs
    run_ids = values.ne(values.shift()).cumsum()
    run_lengths = values.groupby(run_ids).size()

    longest_run = run_lengths.max()

    print(f"{col}: longest consecutive run = {longest_run}")

# --------------------------------------------------
# 6. TARGET ANOMALIES
# --------------------------------------------------
print("\n[6] TARGET ANOMALY INVESTIGATION")

if candidate in train.columns:

    valid = train[[target, candidate]].dropna()
    ratio = valid[target] / valid[candidate]

    print("\nTarget/candidate ratio statistics:")
    print(ratio.describe())

    suspicious = valid[(ratio >= 9) & (ratio <= 11)]

    print("\nRecords with ratio between 9 and 11:")

    print(
        train.loc[
            suspicious.index,
            ["record_id", target, candidate]
        ]
    )

    print("\nSuspicious record count:", len(suspicious))

# --------------------------------------------------
# 7. SPELLING VARIANTS
# --------------------------------------------------
print("\n[7] CATEGORICAL SPELLING VARIANTS")

categorical_cols = ["supply_source", "tower"]

for col in categorical_cols:

    print(f"\n{col} unique values:")
    print(sorted(train[col].dropna().unique()))

    print(f"\n{col} value counts:")
    print(train[col].value_counts(dropna=False))

# --------------------------------------------------
# INVESTIGATION COMPLETE
# --------------------------------------------------
print("\n" + "=" * 60)
print("INVESTIGATION COMPLETED")
print("=" * 60)
