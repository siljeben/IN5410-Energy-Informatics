import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from appliance import Appliance
from household import Household
from neighborhood import Neighborhood
from helper_functions import get_appliances
from eval_functions import plot_household_schedule_appliances


# Assumptions:
"""
    - The EV is not home between 8 and 17 and should be done charging at 7
    - The dishwasher can be run at anytime
    - The washing machine should be done when the person is awake so that it can be put in the dryer
"""

appliance_dict = get_appliances(output_dict=True)
ev = appliance_dict["EV"]
washing_machine = appliance_dict["Laundry machine"]
dishwasher = appliance_dict["Dishwasher"]

our_house = Household("Test house")
our_house.add_appliances([ev, washing_machine, dishwasher])

lonely_neighborhood = Neighborhood("Lonely", pricing="ToU", peak_load=1.5)
lonely_neighborhood.add_households([our_house])
lonely_neighborhood.optimize()

reshaped_schedule = lonely_neighborhood.get_schedule().reshape(-1,24)
print(reshaped_schedule)
plot_household_schedule_appliances(our_house, reshaped_schedule, lonely_neighborhood.pricing)

print("pricing:")
pricing = np.sum(lonely_neighborhood.pricing * reshaped_schedule)
print(round(pricing, 5), "Pricing-units")

