# Sensor Drift Pipeline: Usage Guide

How to use `drift` to find drifting sensors, estimate how far they have drifted, and correct the readings.

## What the pipeline does

It learns from healthy data how your sensor columns normally move together (using PCA). When new data arrives, a row whose columns disagree with those relationships is flagged. The pipeline then estimates which column or columns most likely drifted, by how much, and what the readings should have been.

It does not include the machine-health regression. Feed the corrected data into that step.

## Requirements and files

- Python with pandas, numpy and scikit-learn
- `drift`: the pipeline
- `data.csv`: healthy training data (assumed to contain no drift)
- Optional, for testing: `test_data.csv`, `test_truth.csv`

## Quick start

Clone the repo and cd into it

```python
import pandas as pd
from sensor_drift.drift import *

# load healthy data and train model on it
healthy = load_data("sensor_drift/train_data.csv")
model = learn_normal(healthy)

#load test data and test model
test = load_data("sensor_drift/test_data.csv")
report = check(model, test)              # one report row per test row
print(report[report.is_unusual][["suspect_columns", "n_suspects"]])

corrected = apply_corrections(test, report)
```

## The functions

### `load_data(path)`

Reads a csv and prepares it for the model:

- drops the `rpm` column
- treats blank and whitespace-only cells as missing (NaN)
- converts `bearing_freq_peak_Hz` to numbers
- encodes `condition_label` as LOW=0, MEDIUM=1, HIGH=2; unknown labels (such as `NKNOWN`) become NaN

For the training file this leaves 9 columns.

### `learn(healthy, false_alarm_rate=0.01)`

Learns what normal looks like. It removes rows with empty cells (770 rows become 690 for `data.csv`), puts all columns on the same scale, fits PCA keeping the patterns that explain 95% of the variation (5 of 9 on this data), and sets the alert threshold.

`false_alarm_rate` is the share of healthy rows allowed to cross the threshold. It returns a dictionary called the model. Pass that result, not the function itself, to `check`.

To avoid retraining each time:

```python
import joblib
joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

### `check(model, new_data, max_drifting_columns=3)`

Checks new data against the learned model. For each flagged row it finds the most likely culprit column, removes that column's drift, and looks again. If the row still disagrees, it picks the next culprit, and so on until the row looks consistent or the limit is reached. The limit is also capped by the spare room left by the PCA patterns (columns minus patterns minus 1, which is 3 on this data).

The report has one row per input row, with these columns:

- `disagreement`: how much the row's columns disagree with each other
- `is_unusual`: True if it is above the threshold
- `suspect_columns`: the blamed columns, comma-separated (empty if not flagged)
- `n_suspects`: how many columns were blamed
- `drift_<column>`: one per column, the estimated drift in that column's own units (empty if not blamed)
- `note`: "has empty cells - not scored" for rows with blank cells

Requirements: `new_data` needs the same columns as the training data, and its row labels must be unique. Rows with blank cells are never flagged.

### `apply_corrections(data, report)`

Returns a copy of `data` with each blamed column's estimated drift subtracted. Use it to produce the input for the health regression.

### `decode(codes)`

Turns numbers back into LOW, MEDIUM and HIGH. It accepts a list, array or column, rounds to the nearest label, clips values to the 0 to 2 range, and leaves empty values empty. A pandas column keeps its row numbers.

```python
df["condition_name"] = decode(df["condition_label"])
labels = decode(model_predictions)
```

### `compare(report, truth, sensors=DEFAULT_SENSORS)`

For test data where you know the drift that was injected. `report` and `truth` must have the same number of rows in the same order. It prints and returns `(summary, sensor_errors)`:

- `summary`: `caught`, `missed`, `false_alarms`, `drifting_but_not_scored`
- `sensor_errors`: per sensor, `rows_blamed`, `correctly_blamed`, `wrongly_blamed`, `mean_abs_error`

```python
truth = pd.read_csv("test_truth.csv")
summary, sensor_errors = compare(report, truth)
```

The sensors compared are set in `DEFAULT_SENSORS`, as `sensor column: (truth column, smallest drift that counts as real)`. To compare another sensor, add its true drift as a new column in the truth table, then pass your own dictionary:

```python
my_sensors = {
    **DEFAULT_SENSORS,
    "vibration_rms_g": ("true_vibration_drift", 0.02),
}
compare(report, truth, sensors=my_sensors)
```

## Reading the results

A single flagged row is weak evidence, because drift builds up slowly. Look at runs of consecutive flagged rows, and summarise per column:

```python
bad = report[report.is_unusual]
print(bad.filter(like="drift_").agg(["count", "median"]).T)
```

A genuinely drifting sensor shows up as many flagged rows blaming the same column with a similar drift value.

Correlated columns can be blamed in place of each other. On the generated test set, `ambient_temp_C` and `temperature_sensor_C` both appeared, with drifts of opposite sign, for what was a temperature sensor fault. When two related columns show opposite drifts together, inspect both sensors.

Rows that were not scored can be listed with `report[report.note != ""]`.

## Decoding labels

`condition_label` is stored as numbers (LOW=0, MEDIUM=1, HIGH=2) so the model can use it. `decode` turns numbers back into names, whether they are the original labels or model output.

```python
from final_pipeline import load_data, decode

df = load_data("data.csv")
df["condition_name"] = decode(df["condition_label"])
print(df["condition_name"].value_counts(dropna=False))
```

On `data.csv` this gives 297 LOW, 264 MEDIUM, 208 HIGH and 1 empty value (the `NKNOWN` row).

Model output can be fractional. `decode` rounds to the nearest label and clips values to the 0 to 2 range:

```python
decode([0, 1, 2, 1.2, 2.7, -0.3])   # LOW, MEDIUM, HIGH, MEDIUM, HIGH, LOW
labels = decode(predictions)        # predictions from your own model
```

Empty values stay empty. A pandas column keeps its row numbers, while a list or array is numbered from 0. The corrected data from `apply_corrections` also holds `condition_label` as numbers, so decode that column the same way.

## Comparing against the test truth data

`test_truth.csv` is the answer key for `test_data.csv`: row n of one matches row n of the other. Check the pipeline's report against it with `compare`:

```python
import pandas as pd
from final_pipeline import load_data, learn_normal, check, compare

model = learn_normal(load_data("data.csv"))
report = check(model, load_data("test_data.csv"))
truth = pd.read_csv("test_truth.csv")

summary, sensor_errors = compare(report, truth)
```

On the generated test set the summary is 104 caught, 59 missed, 0 false alarms and 30 drifting rows not scored because of blank cells. Per sensor:

| Sensor | Rows blamed | Correctly blamed | Wrongly blamed | Mean abs. error |
| --- | --- | --- | --- | --- |
| `temperature_sensor_C` | 63 | 63 | 0 | 0.212 (degrees C) |
| `current_sensor_A` | 5 | 5 | 0 | 0.007 (A) |

How to read it:

- `caught`: flagged rows that truly have drift above the sensor's tolerance
- `missed`: scored rows with real drift that were not flagged, mostly early rows where the drift is still small
- `false_alarms`: flagged rows with no real drift
- `drifting_but_not_scored`: rows with real drift that have blank cells
- `mean_abs_error`: the average gap between the estimated and true drift, in the sensor's own units

Only the sensors listed in `DEFAULT_SENSORS` are compared. `ambient_temp_C` was blamed in 41 flagged rows, but it has no truth column, so it does not appear in this table. To compare other sensors, add a truth column for them and pass your own dictionary, as shown under `compare` above.

## Tuning

- `false_alarm_rate`: lower it for fewer alerts, raise it for earlier detection.
- PCA setting: `PCA(n_components=0.95)` inside `learn_normal`. A higher value keeps more patterns, which can hide drift. A lower value leaves more unexplained, which raises false alarms. Other values were not tested on this data, so try a few against injected drifts.
- `max_drifting_columns`: the most columns `check` may blame in one row.

## Known limitations

- Drift in a column that is not predictable from the others (such as `humidity_percent`) cannot be detected, because it is not related to the other detected values
- Small drift is missed. On the generated test set, flagging began once the temperature drift reached roughly 0.4 degrees C. A current drift of 0.06 A went undetected, while 0.1 A was caught.
- With several drifting columns the pipeline flags the rows, but often names the wrong columns. In a test with two drifting sensors (temperature +2, current +0.1) every row was flagged, but the correct pair was named in only 35 of 140 rows. With three drifting sensors, the third was not found. Do not trust corrections from rows with two or more suspects until the same columns are named across many consecutive rows.
- Rows with blank cells are not scored.
- The healthy training data must be genuinely healthy, and should cover the full operating range. Removing rows with blanks removes many low-load startup rows.
- All test results above come from synthetic faults added to your own data. Real faults may be messier.
