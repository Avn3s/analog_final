# Sensor Drift Pipeline: Usage Guide

How to use `drift.py` to find drifting sensors, estimate how far they have drifted, and correct the readings.

## What the pipeline does

It learns from healthy data how your sensor columns normally move together (using PCA). When new data arrives, a row whose columns disagree with those relationships is flagged. The pipeline then estimates which column or columns most likely drifted, by how much, and what the readings should have been.

It does not include the machine-health regression. Feed the corrected data into that step. It also reports a health index for each sensor: the percentage of its readings that were recorded correctly.

## Requirements and files

- Python with pandas, numpy and scikit-learn
- `drift.py`: the pipeline
- `data.csv`: healthy training data (assumed to contain no drift)
- Optional, for testing: `test_data.csv`, `test_truth.csv`

## Quick start
Clone the repo, cd into the repo and install dependencies
```bash
git clone https://github.com/Avn3s/analog_project.git
cd analog_final
#pip users
pip install -r requirements.txt

#uv users
uv pip install -r requirements.txt

```

Create a new file to try out the functions in sensor_drift/drift.py
```python
from sensor_drift.drift import *

print("Healthy Training Data: ")
healthy = load_data("sensor_drift/train_data.csv")
print(healthy)
model = learn(healthy)

print("\n\nInconsistent Data in the Test dataset:")
test = load_data("sensor_drift/test_data.csv")
report = check(model, test)

print(report[report.is_unusual][["suspect_columns", "n_suspects"]])

print("\n\nCorrected Test dataset:")
corrected = apply_corrections(test, report)

print(corrected)

print("\n\nSensor Health:")

sensor_health = sensor_health_index(report)
for sensor in sensor_health:
    print(f"\t- {sensor}_reliability: {sensor_health[sensor]}")

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

`DEFAULT_SENSORS` covers every sensor. Each entry is `sensor column: (truth column, smallest drift that counts as real)`. A sensor with no column in the truth table is treated as having no drift, so any row where it is blamed counts as a wrong blame. The defaults are:

| Sensor | Truth column | Smallest real drift |
| --- | --- | --- |
| `load_percent` | `true_load_percent_drift` | 0.1 |
| `ambient_temp_C` | `true_ambient_temp_C_drift` | 0.05 |
| `temperature_sensor_C` | `true_temp_drift` | 0.05 |
| `humidity_percent` | `true_humidity_percent_drift` | 0.5 |
| `current_sensor_A` | `true_current_drift` | 0.005 |
| `vibration_rms_g` | `true_vibration_rms_g_drift` | 0.03 |
| `vibration_peak_g` | `true_vibration_peak_g_drift` | 0.1 |
| `bearing_freq_peak_Hz` | `true_bearing_freq_peak_Hz_drift` | 15 |

To compare only some sensors, or to change a tolerance, pass your own dictionary:

```python
two_sensors = {
    "temperature_sensor_C": ("true_temp_drift", 0.05),
    "current_sensor_A": ("true_current_drift", 0.005),
}
compare(report, truth, sensors=two_sensors)
```

### `sensor_health_index(report, sensor=None)`

Defined in `drift.py`. Gives the percentage (0 to 100) of a sensor's readings that were recorded correctly. A reading counts as correct when the sensor was not blamed for drift in that row. Rows with empty cells are left out, because they cannot be checked. A value of 100.0 means the sensor was never blamed, and lower values mean it was blamed more often.

Parameters:

- `report`: the report returned by `check()`
- `sensor` (optional): one column name, such as `"temperature_sensor_C"`

Returns:

- with a sensor: a plain float, rounded to one decimal place
- without a sensor: a dictionary of `{sensor: float}` for every sensor, meaning every column except `condition_label`

```python
from drift import load_data, learn, check, sensor_health_index

model = learn(load_data("data.csv"))
report = check(model, load_data("test_data.csv"))

sensor_health_index(report)                            # every sensor
sensor_health_index(report, "temperature_sensor_C")   # 75.4
sensor_health_index(report.tail(100))                  # only the most recent 100 rows
```

On the generated test set (256 of 300 rows could be scored):

| Sensor | Health index |
| --- | --- |
| `temperature_sensor_C` | 75.4 |
| `ambient_temp_C` | 84.0 |
| `current_sensor_A` | 98.0 |
| all other sensors | 100.0 |

On the first 100 rows, which have no drift, every sensor scores 100.0.

Errors:

- `TypeError`: `sensor` is not a single column name (for example a list)
- `ValueError`: `sensor` is not a column of your data, or no rows could be scored

What the index cannot tell you:

- It measures what the pipeline blamed, not what truly drifted. A column that cannot be checked against the others, such as `humidity_percent`, shows 100 even if it drifts.
- Correlated columns can be blamed in place of each other. `ambient_temp_C` scores 84.0 above although only the temperature sensor drifted.
- Drift smaller than the detection level counts as correct.
- Rows with blank cells are ignored, so a sensor that is often blank can still score high.

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
from drift import load_data, decode

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
from drift import load_data, learn, check, compare

model = learn(load_data("data.csv"))
report = check(model, load_data("test_data.csv"))
truth = pd.read_csv("test_truth.csv")

summary, sensor_errors = compare(report, truth)
```

On the generated test set the summary is 104 caught, 59 missed, 0 false alarms and 30 drifting rows not scored because of blank cells. Per sensor:

| Sensor | Rows blamed | Correctly blamed | Wrongly blamed | Mean abs. error |
| --- | --- | --- | --- | --- |
| `ambient_temp_C` | 41 | 0 | 41 | 0.953 (degrees C) |
| `temperature_sensor_C` | 63 | 63 | 0 | 0.212 (degrees C) |
| `current_sensor_A` | 5 | 5 | 0 | 0.007 (A) |
| The other 5 sensors | 0 | 0 | 0 | never blamed |

How to read it:

- `caught`: flagged rows that truly have drift above the sensor's tolerance
- `missed`: scored rows with real drift that were not flagged, mostly early rows where the drift is still small
- `false_alarms`: flagged rows with no real drift
- `drifting_but_not_scored`: rows with real drift that have blank cells
- `mean_abs_error`: the average gap between the estimated and true drift, in the sensor's own units

All sensors in `DEFAULT_SENSORS` are compared. `ambient_temp_C` has no truth column, so it counts as having no drift, and its 41 blamed rows are all wrong blames. This is the mix-up with the temperature sensor described under Reading the results. To compare a sensor that did drift, add a truth column for it, named as in the defaults table above.

## Tuning

- `false_alarm_rate`: lower it for fewer alerts, raise it for earlier detection.
- PCA setting: `PCA(n_components=0.95)` inside `learn`. A higher value keeps more patterns, which can hide drift. A lower value leaves more unexplained, which raises false alarms. Other values were not tested on this data, so try a few against injected drifts.
- `max_drifting_columns`: the most columns `check` may blame in one row.

## Known limitations

- Drift in a column that is not predictable from the others (such as `humidity_percent`) cannot be detected, because it does not disturb the relationships between columns.
- Small drift is missed. On the generated test set, flagging began once the temperature drift reached roughly 0.4 degrees C. A current drift of 0.06 A went undetected, while 0.1 A was caught.
- With several drifting columns the pipeline flags the rows, but often names the wrong columns. In a test with two drifting sensors (temperature +2, current +0.1) every row was flagged, but the correct pair was named in only 35 of 140 rows. With three drifting sensors, the third was not found. Do not trust corrections from rows with two or more suspects until the same columns are named across many consecutive rows.
- Rows with blank cells are not scored.
- The healthy training data must be genuinely healthy, and should cover the full operating range. Removing rows with blanks removes many low-load startup rows.
- All test results above come from synthetic faults added to your own data. Real faults may be messier.

## Troubleshooting

- `'function' object is not subscriptable` in `check`: the first argument is the function, not its result. Use `model = learn(healthy)`, then `check(model, test)`.
- `'numpy.ndarray' object is not callable`: a variable has the same name as a function in the file (for example `unexplained_part`). Rename the variable, and in a notebook restart the kernel and run all cells from the top.
- "duplicate row labels" from `check`, or `InvalidIndexError`: the table has repeated row labels. Use `test = test.reset_index(drop=True)`.
- "report has N rows but truth has M": the two tables must have the same number of rows, in the same order.
- "truth has duplicate column names": a column was added twice or joined twice. Remove the duplicate.
