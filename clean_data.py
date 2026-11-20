from pathlib import Path
import pandas as pd

# Paths
INPUT = Path("data/train.csv")
OUTPUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset (never overwrite it)
df = pd.read_csv(INPUT)
original_rows = len(df)

# 1. Remove exact duplicate rows
df = df.drop_duplicates().copy()
duplicates_removed = original_rows - len(df)

# 2. Handle sentinel value
sentinel_count = (df["supply_hours"] == -999).sum()
df["supply_hours"] = df["supply_hours"].replace(-999, pd.NA)

# 3. Flag suspicious target values for review (do not automatically correct)
ratio = df["litres_per_day"] / df["meter_litres_next_morning"]
target_flags = df.loc[
    ratio.between(9.5, 10.5),
    ["record_id", "date", "tower",
     "meter_litres_next_morning", "litres_per_day"]
].copy()

target_flags["anomaly_type"] = "possible_target_10x_error"

# 4. Flag suspicious Tower A garden values for review
garden_flags = df.loc[
    (df["tower"] == "Tower A") & (df["garden_area_m2"] > 2000),
    ["record_id", "date", "tower", "garden_area_m2"]
].copy()

garden_flags["anomaly_type"] = "possible_garden_area_scaling"

# 5. Save cleaned dataset separately
cleaned_file = OUTPUT_DIR / "train_cleaned.csv"
df.to_csv(cleaned_file, index=False)

# 6. Save anomaly review report
anomaly_report = pd.concat(
    [
        target_flags.assign(column="litres_per_day"),
        garden_flags.assign(column="garden_area_m2"),
    ],
    ignore_index=True,
    sort=False
)

anomaly_file = REPORT_DIR / "phase1_anomaly_flags.csv"
anomaly_report.to_csv(anomaly_file, index=False)

# 7. Summary
print("=" * 45)
print("PHASE 1 CLEANING SUMMARY")
print("=" * 45)
print(f"Original rows              : {original_rows}")
print(f"Exact duplicates removed   : {duplicates_removed}")
print(f"Sentinel values handled   : {sentinel_count}")
print(f"Rows after cleaning       : {len(df)}")
print(f"Target anomaly flags      : {len(target_flags)}")
print(f"Garden anomaly flags      : {len(garden_flags)}")
print(f"Clean dataset            : {cleaned_file}")
print(f"Anomaly report           : {anomaly_file}")

# 8. Validation
assert duplicates_removed == 31, "Unexpected duplicate count!"
assert sentinel_count == 53, "Unexpected sentinel count!"
assert df.duplicated().sum() == 0, "Duplicates remain!"
assert not (df["supply_hours"] == -999).any(), "Sentinel values remain!"

print("\nVALIDATION PASSED")

