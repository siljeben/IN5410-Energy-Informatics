import numpy as np
from datetime import datetime
from PyAstronomy import pyasl

def components_to_angle(df):    
    df["A10"] = np.arctan2(df["U10"], df["V10"])
    df["A100"] = np.arctan2(df["U100"], df["V100"])
    df.drop(['U10', 'V10', 'U100', 'V100'], axis=1)
    return df

def year_decimal_from_timestamp(timestamp: str):
    format_string = "%Y%m%d %H:%M"
    datetime_obj = datetime.strptime(timestamp, format_string)
    return pyasl.decimalYear(datetime_obj)