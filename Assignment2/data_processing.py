import numpy as np
from datetime import datetime


def components_to_angle(df):
    df["A10"] = np.arctan2(df["U10"], df["V10"])
    df["A100"] = np.arctan2(df["U100"], df["V100"])
    df.drop(["U10", "V10", "U100", "V100"], axis=1)
    return df


def convert_str_to_datetime(timestamp: str):
    format_string = "%Y%m%d %H:%M"
    datetime_obj = datetime.strptime(timestamp, format_string)
    return datetime_obj


def get_sliding_window_input_output(data: np.ndarray, window_size: int):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
    return np.array(X), data[window_size:]


def get_sliding_window_input_test_data(
    test: np.ndarray, train: np.ndarray, window_size: int
):
    X = []
    for i in range(window_size):
        x = np.concatenate([train[-window_size + i :], test[:i]])
        X.append(x)
    for i in range(len(test) - window_size):
        X.append(test[i : i + window_size])
    return np.array(X)
