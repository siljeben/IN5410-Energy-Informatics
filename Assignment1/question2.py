import numpy as np
from neighborhood import Neighborhood
from household import Household
from plot_functions import plot_schedule_appliances, plot_schedule_shiftable_nonshiftable
from helper_functions import get_appliances, get_random_optional_shiftable


try: 
    # if the function has run before, load value
    random_household: Household = Household.load('data/random_household.pkl')
except:
    random_household: Household = Household("Our second house")

    # add all shiftable and non-shiftable appliances
    nonshiftable_appliances = get_appliances(filter_shiftable=0)
    shiftable_appliances = get_appliances(filter_shiftable=1)

    # add a random combination of optional appliances
    optional_appliances = get_random_optional_shiftable()

    random_household.add_appliances(nonshiftable_appliances + shiftable_appliances + optional_appliances)
    
    random_household.save('data/random_household.pkl')

finally: 
    lonely_richer_neighborhood = Neighborhood("Another lonely", pricing="RTP")
    lonely_richer_neighborhood.add_households([random_household])
    lonely_richer_neighborhood.optimize()

    plot_schedule_appliances(lonely_richer_neighborhood, include_house_name=False)
    plot_schedule_shiftable_nonshiftable(lonely_richer_neighborhood)

    cost = lonely_richer_neighborhood.optimized_value
    print(f"The energy bill is {cost:.2f} NOK")
