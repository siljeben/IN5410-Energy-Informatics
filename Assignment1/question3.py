import numpy as np
from plot_functions import plot_schedule_appliances, plot_schedule_shiftable_nonshiftable
from neighborhood import Neighborhood


n_households = 30

crowded_neighborhood = Neighborhood(name="New neighborhood", households=n_households, pricing="RTP")
res = crowded_neighborhood.optimize()

plot_schedule_appliances(crowded_neighborhood)
plot_schedule_shiftable_nonshiftable(crowded_neighborhood)

cost = crowded_neighborhood.optimized_value
print(f"The energy bill is {cost:.2f} NOK")