<WritingBlock id="58321" variant="document"># Phase 1 – Data Investigation and Cleaning
1. Dataset Overview
The Housing Society Water Demand dataset contains information about daily water usage in a housing society in Nagpur.
- Total records: 1031
- Total columns: 12
- Target variable: litres_per_day
- Problem type: Regression
The dataset contains information about occupancy, temperature, garden area, tank levels, water supply, complaints, and water consumption.
2. Data Quality Investigation
The dataset was checked for missing values, duplicate records, unusual values, inconsistent units, suspicious target values, and possible data leakage.
Anomalies Identified
- Duplicate rows: 31 exact duplicate records were found.
- Sentinel values: The supply_hours column contained -999, which represents an invalid or missing value.
- Target anomalies: 13 records showed suspicious differences between litres_per_day and meter_litres_next_morning.
- Unit mix-up: Tower A had unusually large garden_area_m2 values, suggesting a possible unit inconsistency.
- Leaking column: meter_litres_next_morning was investigated as a potential leakage feature.
- Stuck sensor: No convincing stuck-sensor pattern was found.
- Spelling variants: No significant spelling inconsistencies were detected in categorical columns.
3. Data Cleaning
The cleaning process was performed using clean_data.py.

The following steps were carried out:
1. Removed 31 exact duplicate rows.
2. Replaced 53 sentinel values (-999) in supply_hours with missing values.
3. Flagged 13 suspicious target records for further investigation.
4. Flagged 272 suspicious garden-area records belonging to Tower A.
5. Preserved suspicious records instead of making unsupported corrections.
4. Final Dataset
After removing duplicates, the dataset contains 1000 records.

The cleaned dataset was saved as:

data/processed/train_cleaned.csv

Anomaly flags were saved as:

reports/phase1_anomaly_flags.csv
5. Validation
The cleaning pipeline was executed successfully.

Validation confirmed that:
- No exact duplicate rows remain.
- No -999 sentinel values remain in supply_hours.
- The cleaned dataset contains 1000 records.
- Target and garden-area anomalies are documented in the anomaly report.
6. Conclusion
Phase 1 data investigation and cleaning established a clean baseline dataset for subsequent model development. Identified anomalies were documented and flagged, while uncertain values were preserved to avoid making unsupported corrections. The processed dataset and investigation results are ready for the next phase of the MLOps workflow.