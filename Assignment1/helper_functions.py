import pandas as pd
from appliance import Appliance
import numpy as np
from matplotlib import pyplot as plt

def get_appliances(filter_shiftable=None, random_selection_n=None, output_dict=False) -> list[Appliance] | dict[str, Appliance]:
    """Function that gets all or n random appliances based on a filter and outputs it as either a dictionary or a list

    Args:
        filter_shiftable (int, optional): Filter for if we only want the non-shiftable, semi-shiftable or the shiftable appliances. Defaults to None.
        random_selection_n (int, optional): Value for selecting n random appliances. Defaults to None.
        output_dict (bool, optional): If true the function returns a dictionary and if false a list. Defaults to False.

    Returns:
        list[Appliance] | dict[str, Appliance]: A list or dict containing appliances.
    """

    if output_dict:
        appliances_dict = {}
    else:
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
        if output_dict:
            appliances_dict[appliance.name] = appliance
        else:
            appliances_list.append(appliance)
    if output_dict:
        return appliances_dict
    return appliances_list

def get_random_optional_shiftable(n=4):
    """randomly drops some optional appliances to simulate a household

    Args:
        n (int, optional): number of random appliances we want. Defaults to 4.

    Returns:
        list[Appliance] | dict[str, Appliance]: A list or dict containing appliances.
    """
    return get_appliances(filter_shiftable=2, random_selection_n=n)

def get_pricing(pricing: str) -> np.ndarray:
    """ A function that fetches the pricing in a numpy array

    Args:
        pricing (str): The type of pricing wanted, either "ToU" (Time-of-Use) or "RTP" (Real-Time-Pricing)

    Raises:
        ValueError: Is raised if value isn't ToU or RTP

    Returns:
        np.ndarray: A numpy array of length 24 containing the pricing data for each hour of the day
    """
    if pricing == "ToU":
        pricing = np.zeros(24)
        pricing[0:17] = 0.5
        pricing[17:20] = 1.0
        pricing[20:24] = 0.5
    elif pricing == "RTP":
        pricing = np.load('data/rt_pricing.npy')
    else:
        raise ValueError("Pricing must be either 'ToU' or 'RTP'.")
    return pricing

if __name__ == "__main__":
    print(get_appliances())
    print(get_appliances(filter_shiftable=2))
    print(get_pricing("ToU"))
    print(get_pricing("RTP"))

    rtp_pricing = get_pricing("RTP")
    
    plot_hours = np.arange(48) / 2
    plot_pricing = np.repeat(rtp_pricing, 2)
    
    plt.plot(plot_hours, plot_pricing, 'r')

    plt.xlim(-1, 24)
    plt.xticks(np.arange(0, 24, 2))
    
    plt.grid()

    plt.xlabel("Time [h]")
    plt.ylabel("Electricity price [NOK/kWh]")

    plt.show()
