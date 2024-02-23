import pandas as pd
import enum
from classes import Appliance

def get_appliances() -> enum.Enum:
    appliances_dict = {}

    df_appliances = pd.read_excel('data/energy_usage.xlsx')
    for i, row in df_appliances.iterrows():
        appliance = Appliance(row['Appliances'].replace(" ", ""),
                              row['Shiftable'],
                              row['Length [h]'],
                              row['Daily usage [kWh]'],
                              row['Alpha'],
                              row['Beta']
        )
        appliances_dict[appliance.name] = appliance
    return appliances_dict

if __name__ == "__main__":
    print(get_appliances()['Laundrymachine'])