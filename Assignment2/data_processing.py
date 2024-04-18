import pandas as pd
import numpy as np

def polar_angle(x, y):
    angle = np.arctan2(y,x)
    angle[angle<0] = 2*np.pi + angle[angle<0]
    return angle

def components_to_angle(df):    
    df["A10"] = polar_angle(df["U10"], df["V10"])
    df["A100"] = polar_angle(df["U100"], df["V100"])
    df.drop(['U10', 'V10', 'U100', 'V100'], axis=1)
    return df

