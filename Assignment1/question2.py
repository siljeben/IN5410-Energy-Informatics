import requests as r
import pandas as pd

def RTP_pricing():
    return pd.read_json(r.get("https://www.hvakosterstrommen.no/api/v1/prices/2023/02-22_NO1.json").text)

pricing = RTP_pricing()

data = pd.read_csv("datafile.csv")




