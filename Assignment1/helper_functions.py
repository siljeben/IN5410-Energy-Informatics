import pandas as pd
from classes import Appliance

def get_appliances(filter_shiftable=None) -> dict[str, Appliance]:

    appliances_dict = {}

    df_appliances = pd.read_excel('data/energy_usage.xlsx')

    if filter_shiftable is not None:
        df_appliances = df_appliances[df_appliances['Shiftable'] == filter_shiftable]

    for i, row in df_appliances.iterrows():
        appliance = Appliance(row['Appliances'],
                              row['Shiftable'],
                              row['Length [h]'],
                              row['Daily usage [kWh]'],
                              row['Alpha'],
                              row['Beta']
        )
        appliances_dict[appliance.name] = appliance
    return appliances_dict

if __name__ == "__main__":
    print(get_appliances())
    print(get_appliances(filter_shiftable=2))
