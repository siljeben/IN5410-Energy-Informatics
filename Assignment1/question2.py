import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# timeslots for pricing

def generate_pricing_data():
    pricing = np.zeros(24)
    pricing[0:17] = np.random.uniform(0.45, 0.65, 17)
    pricing[17:20] = np.random.uniform(0.75, 1.0, 3)
    pricing[20:24] = np.random.uniform(0.45, 0.65, 4)
    np.save('data/rt_pricing.npy', pricing)

# generate_pricing_data()

pricing = np.load('data/rt_pricing.npy')

pricing_plot = np.zeros(48)
pricing_plot[::2] = pricing
pricing_plot[1::2] = pricing
plt.plot(np.arange(48)/2, pricing_plot)
plt.ylim(0, 1)
plt.show()

