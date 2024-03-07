import numpy as np
from helper_functions import get_pricing

from neighborhood import Neighborhood

pricing = get_pricing("RTP")

neighborhood = Neighborhood(name="Neighborhood 1", households=30, pricing="RTP", peak_load=0)

res = neighborhood.optimize()
schedule = neighborhood.get_schedule()
print(f"Length of state vector x: {len(schedule)}")
print(f"# of nonzero elements in state vector: {np.where(schedule!=0)[0].size}")

neighborhood.calc_house_schedules()
print("\n")
print("Example of a house schedule:")
print(neighborhood.house_schedules[0])
