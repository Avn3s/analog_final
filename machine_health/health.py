import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action="ignore")


LABELS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
MODEL = RandomForestClassifier(n_estimators=2000, warm_start=True)


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


def learn(x, y, model=MODEL):
    model.fit(x, y)
    return model


def check(x_pred, model):
    y_pred = model.predict(x_pred)
    machine_health_index = y_pred.mean()
    return machine_health_index
