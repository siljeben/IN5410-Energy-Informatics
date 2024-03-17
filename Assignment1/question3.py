import numpy as np
from helper_functions import get_pricing
from eval_functions import plot_neighborhood_schedule_shiftable_nonshiftable

from neighborhood import Neighborhood

pricing = get_pricing("RTP")

n_households = 30

neighborhood = Neighborhood(name="Neighborhood 1", households=n_households, pricing="RTP", peak_load=0)

res = neighborhood.optimize()
schedule = neighborhood.get_schedule()
print(f"Length of state vector x: {len(schedule)}")
print(f"# of nonzero elements in state vector: {np.where(schedule!=0)[0].size}")

neighborhood.calc_house_schedules()
print("\n")
print("Example of a house schedule:")
print(neighborhood.house_schedules[0])

plot_neighborhood_schedule_shiftable_nonshiftable(neighborhood)

print("pricing:")
pricing_array = [np.multiply(schedules, neighborhood.pricing) for schedules in neighborhood.get_house_schedules()]

total_sum = 0.0
for subarray in pricing_array:
    subarray_sum = np.sum(subarray)
    total_sum += subarray_sum

print("Total sum:", total_sum, "NOK")
print("Sum per house:", total_sum/n_households, "NOK")
