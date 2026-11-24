
## 1. Data leakage

Column: meter_litres_next_morning
Present in train: True
Present in holdout: False
Correlation with target: 0.3037
Rows with target/meter ratio within 5% of 1: 1006/1031
Leakage detected: True

## 2. Garden-area unit mix-up

         count    min          mean       max
tower                                        
Tower A    299  21.53  11360.088127  21474.18
Tower B    473   4.00    968.033827   1998.00
Tower C    259   4.00   1035.602317   1995.00

Reference maximum (Tower B/C): 1998.0
Suspicious Tower A records: 280
Conversion factor (square feet to square metres): 10.764
Converted minimum: 194.00
Converted mean: 1120.81
Converted maximum: 1995.00
Unit mix-up detected: True

## 3. Sentinel value

Column: supply_hours
Value: -999
Count: 54
Present: True

## 4. Duplicate rows

Exact duplicate rows: 31
Present: True

## 5. Stuck sensor

  tower                 column  value start_date   end_date  span_days  observations
Tower B        leak_complaints    5.0 2026-03-25 2026-03-27          3             3
Tower A         flats_occupied   74.0 2026-05-18 2026-05-19          2             2
Tower A         flats_occupied  293.0 2026-06-07 2026-06-08          2             2
Tower A        leak_complaints    2.0 2026-02-03 2026-02-04          2             2
Tower A        leak_complaints    4.0 2026-02-12 2026-02-13          2             2
Tower A        leak_complaints    3.0 2026-04-05 2026-04-06          2             2
Tower A        leak_complaints    9.0 2026-05-19 2026-05-20          2             2
Tower B tank_level_morning_pct   15.0 2026-05-12 2026-05-13          2             2
Tower B        leak_complaints    2.0 2026-01-13 2026-01-14          2             2
Tower B        leak_complaints    4.0 2026-01-18 2026-01-19          2             2
Tower B        leak_complaints    1.0 2026-01-30 2026-01-31          2             2
Tower B        leak_complaints   10.0 2026-02-14 2026-02-15          2             2
Tower B        leak_complaints    6.0 2026-02-17 2026-02-18          2             2
Tower B        leak_complaints    6.0 2026-02-23 2026-02-24          2             2
Tower B        leak_complaints    2.0 2026-03-05 2026-03-06          2             2
Tower B        leak_complaints    1.0 2026-03-11 2026-03-12          2             2
Tower B        leak_complaints    9.0 2026-04-08 2026-04-09          2             2
Tower B        leak_complaints    7.0 2026-04-23 2026-04-24          2             2
Tower B        leak_complaints    8.0 2026-04-27 2026-04-28          2             2
Tower B        leak_complaints    2.0 2026-05-17 2026-05-18          2             2
Tower B        leak_complaints    3.0 2026-05-24 2026-05-25          2             2
Tower B        leak_complaints   10.0 2026-05-29 2026-05-30          2             2
Tower B        leak_complaints    2.0 2026-06-11 2026-06-12          2             2
Tower B        leak_complaints   10.0 2026-06-21 2026-06-22          2             2
Tower B        leak_complaints    3.0 2026-06-23 2026-06-24          2             2
Tower B           supply_hours    3.0 2026-02-13 2026-02-14          2             2
Tower B           supply_hours    1.4 2026-04-11 2026-04-12          2             2
Tower C         flats_occupied   41.0 2026-03-05 2026-03-06          2             2
Tower C tank_level_morning_pct   16.0 2026-04-20 2026-04-21          2             2
Tower C tank_level_morning_pct   29.0 2026-04-29 2026-04-30          2             2
Tower C tank_level_morning_pct   38.0 2026-06-17 2026-06-18          2             2
Tower C        leak_complaints    0.0 2026-02-10 2026-02-11          2             2
Tower C        leak_complaints   12.0 2026-04-09 2026-04-10          2             2
Tower C        leak_complaints    8.0 2026-04-11 2026-04-12          2             2
Tower C        leak_complaints    0.0 2026-04-14 2026-04-15          2             2
Tower C        leak_complaints    0.0 2026-04-22 2026-04-23          2             2
Tower C        leak_complaints    5.0 2026-04-24 2026-04-25          2             2
Tower C        leak_complaints   14.0 2026-05-12 2026-05-13          2             2
Tower C        leak_complaints    2.0 2026-06-09 2026-06-10          2             2
Tower C           supply_hours    4.2 2026-01-04 2026-01-05          2             2

Longest repeated run: 3 days
First date: 2026-03-25
Last date: 2026-03-27
Runs of at least 14 days: 0

## 6. Target typos

Largest target gap: 118422
Gap boundary: 107889.00
Values above boundary: 13
All flagged target/meter ratios between 9.5 and 10.5: True

record_id       date   tower  meter_litres_next_morning  litres_per_day  target_meter_ratio
   R00017 2026-01-03 Tower B                      20160          199890            9.915179
   R00029 2026-01-05 Tower C                      25112          253290           10.086413
   R00187 2026-02-06 Tower B                      47731          474260            9.936100
   R00219 2026-02-10 Tower A                      30424          305890           10.054233
   R00287 2026-02-25 Tower C                      22136          217910            9.844145
   R00458 2026-03-27 Tower A                      20200          200670            9.934158
   R00502 2026-04-03 Tower C                      27326          275400           10.078314
   R00559 2026-04-13 Tower A                      19413          190740            9.825375
   R00736 2026-05-15 Tower A                      29695          296700            9.991581
   R00764 2026-05-20 Tower B                      28459          280170            9.844689
   R00979 2026-06-27 Tower C                      43558          428810            9.844575
   R00981 2026-06-27 Tower C                      37868          379410           10.019277
   R00983 2026-06-28 Tower B                      16911          167100            9.881142

## 7. Spelling variants


Column: supply_source
Train value counts:
supply_source
tanker       377
municipal    370
borewell     284
Later value counts:
supply_source
tanker       114
municipal    110
borewell      76
Train-only values: []
Normalized spelling groups: {}
Spelling variants detected: False
True levels in later data: 3

Column: tower
Train value counts:
tower
Tower B    473
Tower A    299
Tower C    259
Later value counts:
tower
Tower B    162
Tower C     73
Tower A     65
Train-only values: []
Normalized spelling groups: {}
Spelling variants detected: False
True levels in later data: 3

## FINAL SUMMARY

TRAIN ROWS: 1031
LATER ROWS: 300
HOLDOUT ROWS: 250

1. LEAKING COLUMN: True
   Column: meter_litres_next_morning
2. UNIT MIX-UP: True
   Column: garden_area_m2
   Group: Tower A
   Factor: 10.764
3. SENTINEL VALUE: True
   Count: 54
4. DUPLICATE ROWS: True
   Count: 31
5. STUCK SENSOR: False
6. TARGET TYPOS: True
   Count: 13
7. SPELLING VARIANTS: False
