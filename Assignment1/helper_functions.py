import pandas as pd
from classes import Appliance

def get_appliances(filter_shiftable=None, random_selection_n=None) -> list[Appliance]:

    appliances_list = []

    df_appliances = pd.read_excel('data/energy_usage.xlsx')

    if filter_shiftable is not None:
        df_appliances = df_appliances[df_appliances['Shiftable'] == filter_shiftable]
    
    if random_selection_n is not None:
        df_appliances = df_appliances.sample(n=random_selection_n)

    for i, row in df_appliances.iterrows():
        appliance = Appliance(row['Appliances'],
                              row['Shiftable'],
                              row['Length [h]'],
                              row['Daily usage [kWh]'],
                              row['Alpha'],
                              row['Beta']
        )
        appliances_list.append(appliance)
    return appliances_list

def get_random_optional_shiftable():
    # randomly drops some optional appliances to simulate a household
    return get_appliances(filter_shiftable=2, random_selection_n=4)

if __name__ == "__main__":
    print(get_appliances())
    print(get_appliances(filter_shiftable=2))
