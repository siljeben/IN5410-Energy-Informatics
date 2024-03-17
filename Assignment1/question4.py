from neighborhood import Neighborhood
from household import Household
from eval_functions import plot_household_schedule_appliances, plot_household_schedule_shiftable_nonshiftable
import numpy as np

random_neighborhood = Neighborhood("another lonely", pricing="RTP", peak_load=1.77005)

# random_neighborhood.add_random_households(1)
# random_neighborhood.houses[0].save('data/random_household.pkl')

random_household = Household.load('data/random_household.pkl')

print(random_household.n_appliances)
print(len(random_household.appliances))

print(random_household.appliances)
random_neighborhood.add_households([random_household])

c, u, l, A_eq, b_eq, A_ub, b_ub = random_neighborhood.get_linprog_input()
res = random_neighborhood.optimize()
schedule = random_neighborhood.get_schedule()

house_schedules = random_neighborhood.get_house_schedules()

with np.printoptions(threshold=np.inf):
    print(np.sum(house_schedules[0], axis=0))

plot_household_schedule_appliances(random_neighborhood.houses[0], house_schedules[0], random_neighborhood.pricing)
plot_household_schedule_shiftable_nonshiftable(random_neighborhood.houses[0], house_schedules[0], random_neighborhood.pricing)

print("pricing:")
pricing = np.sum(random_neighborhood.pricing * house_schedules[0].reshape(-1, 24))
print(round(pricing, 5), "Pricing-units")
