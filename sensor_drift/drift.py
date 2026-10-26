import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LABELS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}



def load_data(path):
    """Read csv data, drop rpm column and rows with empty values,
    and encode every non-numerical value into numerical values."""
    df = pd.read_csv(path, na_values=[" ", ""], skipinitialspace=True)
    df = df.drop(columns=["rpm"], errors="ignore")
    df["bearing_freq_peak_Hz"] = pd.to_numeric(
        df["bearing_freq_peak_Hz"], errors="coerce"
    )
    df["condition_label"] = df["condition_label"].str.strip().str.upper().map(LABELS)
    return df


def decode(codes):
    """Turn numbers (0, 1, 2) back into 'LOW', 'MEDIUM', 'HIGH'."""
    names = {number: name for name, number in LABELS.items()}
    codes = pd.Series(codes, dtype=float)  # keeps the index if you pass a series
    return np.clip(np.rint(codes), 0, 2).map(names)  # empty (NaN) values stay empty


def unexplained_part(model, scaled):
    rebuilt = model["pca"].inverse_transform(model["pca"].transform(scaled))
    return scaled - rebuilt


def learn(healthy, false_alarm_rate=0.01):
    healthy = healthy.dropna()

    # Scale down every column so none dominate
    scaler = StandardScaler().fit(healthy)
    scaled = scaler.transform(healthy)

    # Making model
    pca = PCA(n_components=0.95).fit(scaled)

    # Inconsistency that remains
    unexplained_share = 1 - (pca.components_**2).sum(axis=0)

    model = {
        "columns": list(healthy.columns),
        "scaler": scaler,
        "pca": pca,
        "unexplained_share": unexplained_share,
    }

    # Data is called inconsistent ONLY when errors exceed
    # the throshold defined (see DEFAULT_SENSORS for reference)
    leftover = unexplained_part(model, scaled)
    disagreement = (leftover**2).sum(axis=1)
    model["threshold"] = np.quantile(disagreement, 1 - false_alarm_rate)
    return model


def check(model, new_data, max_drifting_columns=3):
    """For every row, find:
    1. Is there drift at all?
    2. If there is drift, which column(s)
       aka sensor(s) has the drift
    3. How much is the drift?
    4. Wat are the corrected readings?
    """
    columns = model["columns"]
    share = model["unexplained_share"]
    new_data = new_data[columns]
    if not new_data.index.is_unique:
        raise ValueError(
            "new_data has duplicate row labels - use new_data.reset_index(drop=True) first"
        )

    spare_room = len(columns) - model["pca"].n_components_ - 1
    max_columns = max(1, min(max_drifting_columns, spare_room))

    report = pd.DataFrame(
        index=new_data.index,
        columns=["disagreement", "is_unusual", "suspect_columns", "n_suspects"],
    )
    report["is_unusual"] = False
    report["suspect_columns"] = ""
    report["n_suspects"] = 0

    # rows with empty data give unverifyiable positives; theyre being deleted
    has_all_values = new_data.notna().all(axis=1)
    report["note"] = np.where(has_all_values, "", "has empty cells - not scored")
    rows = new_data[has_all_values]
    scaled = model["scaler"].transform(rows)

    # find drift
    leftover = unexplained_part(model, scaled)
    disagreement = (leftover**2).sum(axis=1)
    is_unusual = disagreement > model["threshold"]
    report.loc[rows.index, "disagreement"] = disagreement
    report.loc[rows.index, "is_unusual"] = is_unusual

    # Columns with drift
    fixed = scaled.copy()  # readings with the drift found so far removed
    drift = np.zeros_like(scaled)  # drift found so far (scaled units)
    blamed = np.zeros(
        scaled.shape, dtype=bool
    )  # which columns have been blamed, per row
    still_bad = is_unusual.copy()  # rows that still look inconsistent

    for _ in range(max_columns):
        todo = np.where(still_bad)[0]
        if len(todo) == 0:
            break

        leftover = unexplained_part(model, fixed[todo])
        blame = leftover**2 / share  # column with most blame
        blame[blamed[todo]] = -1  # avoid re-blaming same column
        culprit = blame.argmax(axis=1)

        step = leftover[np.arange(len(todo)), culprit] / share[culprit]
        fixed[todo, culprit] -= step  # correct the drift
        drift[todo, culprit] += step
        blamed[todo, culprit] = True

        now = (unexplained_part(model, fixed[todo]) ** 2).sum(axis=1)
        still_bad[todo] = now > model["threshold"]  # done once the row looks consistent

    # corrected readings
    drift = drift * model["scaler"].scale_
    drift_table = pd.DataFrame(
        np.where(blamed, drift, np.nan),
        index=rows.index,
        columns=["drift_" + c for c in columns],
    )
    report.loc[rows.index, "n_suspects"] = blamed.sum(axis=1)
    report.loc[rows.index, "suspect_columns"] = [
        ", ".join(np.array(columns)[row]) for row in blamed
    ]
    return pd.concat([report, drift_table.reindex(report.index)], axis=1)


def apply_corrections(data, report):
    """Return a copy of `data` with each blamed column's estimated drift removed."""
    corrected = data.copy()
    for column in data.columns:
        if "drift_" + column in report.columns:
            corrected[column] = data[column] - report["drift_" + column].astype(
                float
            ).fillna(0)
    return corrected


DEFAULT_SENSORS = {
    "temperature_sensor_C": ("true_temp_drift", 0.05),
    "current_sensor_A": ("true_current_drift", 0.005),
}


def compare(report, truth, sensors=DEFAULT_SENSORS):
    """Compare check()'s report with the true drift values.
    Prints a summary and returns (summary, sensor_errors)."""
    # Line the two tables up by position (row 1 with row 1, and so on)
    if len(report) != len(truth):
        raise ValueError(
            f"report has {len(report)} rows but truth has {len(truth)} - they must match"
        )
    if truth.columns.duplicated().any():
        raise ValueError(
            "truth has duplicate column names: "
            + str(list(truth.columns[truth.columns.duplicated()]))
        )
    truth = truth.drop(columns=["row"], errors="ignore").reset_index(drop=True)
    report = report.reset_index(drop=True)

    flagged = report["is_unusual"].astype(bool)
    scored = report["note"] == ""  # rows without blank cells

    # Mark row as drifting or inconsistent if drift/error exceeds threshold
    drifting = {}
    truly_drifting = pd.Series(False, index=report.index)
    for sensor, (truth_column, tolerance) in sensors.items():
        drifting[sensor] = truth[truth_column].abs() > tolerance
        truly_drifting |= drifting[sensor]

    summary = {
        "caught": int((flagged & truly_drifting).sum()),
        "missed": int((~flagged & truly_drifting & scored).sum()),
        "false_alarms": int((flagged & ~truly_drifting).sum()),
        "drifting_but_not_scored": int((~scored & truly_drifting).sum()),
    }

    # How close is the estimated drift to true drift?
    sensor_results = {}
    for sensor, (truth_column, tolerance) in sensors.items():
        drift_column = "drift_" + sensor
        if drift_column not in report.columns:
            raise ValueError(
                f"report has no '{drift_column}' column - is '{sensor}' a column of your data?"
            )
        blamed = flagged & report[drift_column].notna()
        error = (
            report.loc[blamed, drift_column].astype(float)
            - truth.loc[blamed, truth_column]
        )
        sensor_results[sensor] = {
            "rows_blamed": int(blamed.sum()),
            "correctly_blamed": int((blamed & drifting[sensor]).sum()),
            "wrongly_blamed": int((blamed & ~drifting[sensor]).sum()),
            "mean_abs_error": round(error.abs().mean(), 3) if blamed.any() else None,
        }
    sensor_errors = pd.DataFrame(sensor_results).T

    print("Summary:", summary)
    print(sensor_errors)
    return summary, sensor_errors
