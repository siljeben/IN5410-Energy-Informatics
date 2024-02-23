import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def generate_pricing_data():
    pricing = np.zeros(24)
    pricing[0:17] = np.random.uniform(0.45, 0.65, 17)
    pricing[17:20] = np.random.uniform(0.75, 1.0, 3)
    pricing[20:24] = np.random.uniform(0.45, 0.65, 4)
    np.save('data/rt_pricing.npy', pricing)

# generate_pricing_data()
pricing = np.load('data/rt_pricing.npy')
# pricing_plot = np.zeros(48)
# pricing_plot[::2] = pricing
# pricing_plot[1::2] = pricing
# plt.plot(np.arange(48)/2, pricing_plot)
# plt.ylim(0, 1)
# plt.show()

def generate_household_appliances(df_appliances):
    # randomly drops some optional appliances to simulate a household
    df_appliances = df_appliances.copy()
    df_shiftable = df_appliances[df_appliances['Shiftable']==2]
    n_shiftable = len(df_shiftable.index)
    n_select = 4
    drop_index = df_shiftable.index[np.random.choice(n_shiftable, n_shiftable - n_select, replace=False)]
    df_appliances = df_appliances.drop(drop_index)
    df_appliances.reset_index(drop=True, inplace=True)
    df_appliances.to_csv('data/energy_usage_selection.csv', index=False)

# df_appliances = pd.read_excel('data/energy_usage.xlsx')
# generate_household_appliances(df_appliances)
df_appliances = pd.read_csv('data/energy_usage_selection.csv')
