import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# Assumptions:
"""
    - The EV is not home between 8 and 17 and should be done charging at 7
    - The dishwasher can be run at anytime
    - The washing machine should be done when the person is awake so that it can be put in the dryer
"""

# timeslots for pricing
pricing = np.zeros(24)
pricing[0:17] = np.random.uniform(0.45, 0.65, 17)
pricing[17:20] = np.random.uniform(0.75, 1.0, 3)
pricing[20:24] = np.random.uniform(0.45, 0.65, 4)


pricing_plot = np.zeros(48)
pricing_plot[::2] = pricing
pricing_plot[1::2] = pricing
plt.plot(np.arange(48)/2, pricing_plot)
plt.ylim(0, 1)
plt.show()
