from household import Household
from neighborhood import Neighborhood
from helper_functions import get_appliances
from plot_functions import plot_schedule_appliances, plot_schedule_shiftable_nonshiftable


# appliances
appliance_dict = get_appliances(output_dict=True)
ev = appliance_dict["EV"]
washing_machine = appliance_dict["Laundry machine"]
dishwasher = appliance_dict["Dishwasher"]

# add appliances to the house
our_house = Household("Our first house")
our_house.add_appliances([ev, washing_machine, dishwasher])

# add house to neighborhood and optimize
lonely_neighborhood = Neighborhood("Lonely", pricing="ToU")
lonely_neighborhood.add_households([our_house])
lonely_neighborhood.optimize()

# get results from optimization
cost = lonely_neighborhood.optimized_value
plot_schedule_appliances(lonely_neighborhood, include_house_name=False)
#plot_schedule_shiftable_nonshiftable(lonely_neighborhood)
print(f"The energy bill is {cost:.2f} NOK")