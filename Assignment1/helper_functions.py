import pandas as pd
import enum
from classes import Appliance

def get_appliances_enum() -> enum.Enum:
    appliances_dict = {}

    df_appliances = pd.read_excel('data/energy_usage.xlsx')
    for i, row in df_appliances.iterrows():
        appliance = Appliance(row['Appliances'].replace(" ", ""),
                              row['Shiftable'],
                            #   row['Length [h]'],
                              row['Daily usage [kWh]'],
                              row['Length [h]'],
                              row['Alpha'],
                              row['Beta']
        )
        appliances_dict[appliance.name] = appliance
    return enum.Enum('Appliances', appliances_dict)

if __name__ == "__main__":
    print(get_appliances_enum())
    print(get_appliances_enum().Laundrymachine)