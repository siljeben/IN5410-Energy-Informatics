import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from classes import Household, Neighborhood, Appliance
from helper_functions import get_appliances


# Assumptions:
"""
    - The EV is not home between 8 and 17 and should be done charging at 7
    - The dishwasher can be run at anytime
    - The washing machine should be done when the person is awake so that it can be put in the dryer
"""

# # timeslots for pricing
# pricing = np.zeros(24)
# pricing[0:17] = 0.5
# pricing[17:20] = 1.0
# pricing[20:24] = 0.5


# pricing_plot = np.zeros(48)
# pricing_plot[::2] = pricing
# pricing_plot[1::2] = pricing
# plt.plot(np.arange(48)/2, pricing_plot)
# plt.ylim(0, 1.3)
# plt.show()

appliance_dict = get_appliances()
ev = appliance_dict["EV"]
washing_machine = appliance_dict["Laundry machine"]
dishwasher = appliance_dict["Dishwasher"]

our_house = Household("Test house")
our_house.add_appliances([ev, washing_machine, dishwasher])

lonely_neighborhood = Neighborhood("Lonely", pricing="ToU")
lonely_neighborhood.add_households([our_house])
#x = lonely_neighborhood.get_linprog_input()
#print(x)
lonely_neighborhood.optimize()
print(lonely_neighborhood.get_schedule().reshape(-1,24))