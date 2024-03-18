from neighborhood import Neighborhood
from household import Household
from plot_functions import plot_schedule_appliances, plot_schedule_shiftable_nonshiftable


try: 
    random_household = Household.load('data/random_household.pkl')

except: 
    raise AssertionError("Question 2 must be run first")

else:
    neighborhood_peak = Neighborhood("Neighborhood with peak", pricing="RTP", peak_load=1.78)
    neighborhood_peak.add_households([random_household])

    res = neighborhood_peak.optimize()

    plot_schedule_appliances(neighborhood_peak, include_house_name=False)
    plot_schedule_shiftable_nonshiftable(neighborhood_peak)

    cost = neighborhood_peak.optimized_value
    print(f"The energy bill is {cost:.2f} NOK")