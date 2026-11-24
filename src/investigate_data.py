
from pathlib import Path
import re
import pandas as pd

# --------------------------------------------------
# 0. PATHS AND DATA
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

TRAIN = DATA_DIR / "train.csv"
LATER = DATA_DIR / "later.csv"
HOLDOUT = DATA_DIR / "holdout_features.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(TRAIN)
later = pd.read_csv(LATER)
holdout = pd.read_csv(HOLDOUT)

report = []


def add(title, text):
    report.append(f"\n## {title}\n\n{str(text)}")


# --------------------------------------------------
# 1. DATA LEAKAGE
# --------------------------------------------------
leak_col = "meter_litres_next_morning"
target_col = "litres_per_day"

leak_present = leak_col in train.columns
leak_in_holdout = leak_col in holdout.columns

if leak_present and target_col in train.columns:
    correlation = train[leak_col].corr(train[target_col])

    ratio = train[target_col] / train[leak_col]
    close_count = ((ratio - 1).abs() <= 0.05).sum()

    leakage_present = not leak_in_holdout and close_count > 0

    leakage_text = (
        f"Column: {leak_col}\n"
        f"Present in train: {leak_present}\n"
        f"Present in holdout: {leak_in_holdout}\n"
        f"Correlation with target: {correlation:.4f}\n"
        f"Rows with target/meter ratio within 5% of 1: "
        f"{close_count}/{len(train)}\n"
        f"Leakage detected: {leakage_present}"
    )
else:
    leakage_present = False
    leakage_text = "Required leakage or target column not found."

add("1. Data leakage", leakage_text)


# --------------------------------------------------
# 2. GARDEN AREA UNIT MIX-UP
# --------------------------------------------------
garden_col = "garden_area_m2"
unit_factor = 10.764

garden_present = False
garden_group = None

if garden_col in train.columns and "tower" in train.columns:

    garden_stats = train.groupby("tower")[garden_col].agg(
        ["count", "min", "mean", "max"]
    )

    garden_text = garden_stats.to_string()

    # Tower B and C are reference groups.
    reference_max = train.loc[
        train["tower"].isin(["Tower B", "Tower C"]),
        garden_col
    ].max()

    tower_a = train.loc[
        train["tower"] == "Tower A", garden_col
    ]

    # Values above the clean reference range are suspicious.
    suspicious_a = tower_a[tower_a > reference_max]

    garden_present = len(suspicious_a) > 0
    garden_group = "Tower A" if garden_present else None

    converted = suspicious_a / unit_factor

    garden_text += (
        f"\n\nReference maximum (Tower B/C): {reference_max}\n"
        f"Suspicious Tower A records: {len(suspicious_a)}\n"
        f"Conversion factor (square feet to square metres): "
        f"{unit_factor}\n"
        f"Converted minimum: {converted.min():.2f}\n"
        f"Converted mean: {converted.mean():.2f}\n"
        f"Converted maximum: {converted.max():.2f}\n"
        f"Unit mix-up detected: {garden_present}"
    )
else:
    garden_text = "Garden area or tower column not found."

add("2. Garden-area unit mix-up", garden_text)


# --------------------------------------------------
# 3. SENTINEL VALUE
# --------------------------------------------------
sentinel_col = "supply_hours"
sentinel_value = -999

if sentinel_col in train.columns:
    sentinel_count = int(
        train[sentinel_col].value_counts(dropna=False)
        .get(sentinel_value, 0)
    )
else:
    sentinel_count = 0

sentinel_present = sentinel_count > 0

add(
    "3. Sentinel value",
    f"Column: {sentinel_col}\n"
    f"Value: {sentinel_value}\n"
    f"Count: {sentinel_count}\n"
    f"Present: {sentinel_present}"
)


# --------------------------------------------------
# 4. DUPLICATE ROWS
# --------------------------------------------------
duplicate_count = int(train.duplicated().sum())
duplicates_present = duplicate_count > 0

add(
    "4. Duplicate rows",
    f"Exact duplicate rows: {duplicate_count}\n"
    f"Present: {duplicates_present}"
)


# --------------------------------------------------
# 5. STUCK SENSOR
# --------------------------------------------------
sensor_columns = [
    "flats_occupied",
    "temp_c",
    "garden_area_m2",
    "tank_level_morning_pct",
    "leak_complaints",
    "supply_hours",
    "meter_litres_next_morning",
]

sensor_columns = [
    c for c in sensor_columns if c in train.columns
]

sensor_data = train.copy()
sensor_data["date"] = pd.to_datetime(
    sensor_data["date"], errors="coerce"
)
sensor_data = sensor_data.dropna(subset=["date"])

# One representative reading per tower per day.
daily = (
    sensor_data
    .sort_values("date")
    .groupby(["tower", "date"])[sensor_columns]
    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
)

stuck_runs = []

for tower in daily.index.get_level_values("tower").unique():

    tower_data = daily.loc[tower].sort_index()

    for column in sensor_columns:

        values = tower_data[column]
        dates = tower_data.index

        start = 0

        for i in range(1, len(values) + 1):

            same = (
                i < len(values)
                and pd.notna(values.iloc[i])
                and pd.notna(values.iloc[i - 1])
                and values.iloc[i] == values.iloc[i - 1]
                and (dates[i] - dates[i - 1]).days == 1
                and values.iloc[i] != sentinel_value
            )

            if same:
                continue

            end = i - 1
            observations = end - start + 1

            if observations >= 2:
                span = (dates[end] - dates[start]).days + 1

                stuck_runs.append({
                    "tower": tower,
                    "column": column,
                    "value": values.iloc[start],
                    "start_date": str(dates[start].date()),
                    "end_date": str(dates[end].date()),
                    "span_days": span,
                    "observations": observations,
                })

            start = i

stuck_df = pd.DataFrame(stuck_runs)

if not stuck_df.empty:
    stuck_df = stuck_df.sort_values(
        ["span_days", "observations"],
        ascending=False
    )

    longest_run = stuck_df.iloc[0]
    stuck_text = stuck_df.to_string(index=False)

    # Handbook describes a sensor stuck for a few weeks.
    stuck_threshold = 14
    long_runs = stuck_df[
        stuck_df["span_days"] >= stuck_threshold
    ]

    stuck_present = not long_runs.empty

    stuck_text += (
        f"\n\nLongest repeated run: "
        f"{longest_run['span_days']} days\n"
        f"First date: {longest_run['start_date']}\n"
        f"Last date: {longest_run['end_date']}\n"
        f"Runs of at least {stuck_threshold} days: "
        f"{len(long_runs)}"
    )
else:
    stuck_present = False
    stuck_text = "No repeated daily readings found."

add("5. Stuck sensor", stuck_text)


# --------------------------------------------------
# 6. TARGET TYPOS
# --------------------------------------------------
target_sorted = train[target_col].dropna().sort_values().reset_index(
    drop=True
)

gaps = target_sorted.diff()

if len(gaps) > 1:
    largest_gap_index = int(gaps.idxmax())
    largest_gap = float(gaps.max())

    # Values above the largest empty gap form the upper group.
    gap_boundary = (
        target_sorted.iloc[largest_gap_index - 1]
        + target_sorted.iloc[largest_gap_index]
    ) / 2

    target_flags = train.loc[
        train[target_col] > gap_boundary,
        [
            "record_id",
            "date",
            "tower",
            "meter_litres_next_morning",
            target_col,
        ],
    ].copy()

    target_flags["target_meter_ratio"] = (
        target_flags[target_col]
        / target_flags["meter_litres_next_morning"]
    )

    target_count = len(target_flags)
    target_present = target_count > 0

    target_text = (
        f"Largest target gap: {largest_gap:.0f}\n"
        f"Gap boundary: {gap_boundary:.2f}\n"
        f"Values above boundary: {target_count}\n"
        f"All flagged target/meter ratios between 9.5 and 10.5: "
        f"{target_flags['target_meter_ratio'].between(9.5, 10.5).all()}\n\n"
        + target_flags.to_string(index=False)
    )
else:
    target_flags = pd.DataFrame()
    target_count = 0
    target_present = False
    target_text = "Not enough target values to identify a gap."

add("6. Target typos", target_text)

target_flags.to_csv(
    REPORT_DIR / "phase1_target_anomaly_check.csv",
    index=False
)


# --------------------------------------------------
# 7. SPELLING VARIANTS
# --------------------------------------------------
text_columns = [
    c for c in train.select_dtypes(include="object").columns
    if c not in ["record_id", "date"]
]

category_results = []
spelling_findings = []

for column in text_columns:

    train_counts = train[column].value_counts(dropna=False)
    later_counts = later[column].value_counts(dropna=False)

    train_levels = set(train[column].dropna().astype(str))
    later_levels = set(later[column].dropna().astype(str))

    # Compare normalized spellings.
    def normalize(value):
        return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

    normalized_groups = {}

    for value in train_levels:
        normalized_groups.setdefault(
            normalize(value), []
        ).append(value)

    duplicate_spellings = {
        key: values
        for key, values in normalized_groups.items()
        if len(values) > 1
    }

    train_only = sorted(train_levels - later_levels)

    has_variants = bool(duplicate_spellings or train_only)

    category_results.append(
        f"\nColumn: {column}\n"
        f"Train value counts:\n{train_counts.to_string()}\n"
        f"Later value counts:\n{later_counts.to_string()}\n"
        f"Train-only values: {train_only}\n"
        f"Normalized spelling groups: {duplicate_spellings}\n"
        f"Spelling variants detected: {has_variants}\n"
        f"True levels in later data: {len(later_levels)}"
    )

    if has_variants:
        spelling_findings.append({
            "column": column,
            "true_levels": len(later_levels),
        })

spelling_present = len(spelling_findings) > 0

add(
    "7. Spelling variants",
    "\n".join(category_results)
    if category_results else "No categorical text columns found."
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------
summary = (
    f"TRAIN ROWS: {len(train)}\n"
    f"LATER ROWS: {len(later)}\n"
    f"HOLDOUT ROWS: {len(holdout)}\n\n"
    f"1. LEAKING COLUMN: {leak_present}\n"
    f"   Column: {leak_col}\n"
    f"2. UNIT MIX-UP: {garden_present}\n"
    f"   Column: {garden_col}\n"
    f"   Group: {garden_group}\n"
    f"   Factor: {unit_factor}\n"
    f"3. SENTINEL VALUE: {sentinel_present}\n"
    f"   Count: {sentinel_count}\n"
    f"4. DUPLICATE ROWS: {duplicates_present}\n"
    f"   Count: {duplicate_count}\n"
    f"5. STUCK SENSOR: {stuck_present}\n"
    f"6. TARGET TYPOS: {target_present}\n"
    f"   Count: {target_count}\n"
    f"7. SPELLING VARIANTS: {spelling_present}\n"
)

add("FINAL SUMMARY", summary)

report_path = REPORT_DIR / "phase1_data_investigation.md"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("=" * 60)
print("SEVEN DATA PROBLEM CHECKS COMPLETED")
print("=" * 60)
print(summary)

print(f"\nDetailed report: {report_path}")
print(
    "Target anomaly report: "
    f"{REPORT_DIR / 'phase1_target_anomaly_check.csv'}"
)
