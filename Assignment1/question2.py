import numpy as np
from neighborhood import Neighborhood
from household import Household

def generate_pricing_data():
    pricing = np.zeros(24)
    pricing[0:17] = np.random.uniform(0.45, 0.65, 17)
    pricing[17:20] = np.random.uniform(0.75, 1.0, 3)
    pricing[20:24] = np.random.uniform(0.45, 0.65, 4)
    np.save('data/rt_pricing.npy', pricing)

random_neighborhood = Neighborhood("another lonely", pricing="RTP")

# random_neighborhood.add_random_households(1)
# random_neighborhood.houses[0].save('data/random_household.pkl')

random_household = Household.load('data/random_household.pkl')
print(random_household.appliances)
random_neighborhood.add_households([random_household])

c, u, l, A_eq, b_eq, A_ub, b_ub = random_neighborhood.get_linprog_input()
#print(A_eq, b_eq)
random_neighborhood.optimize()

print(random_neighborhood.houses[0])
print(random_neighborhood.get_schedule().reshape(-1,24))
