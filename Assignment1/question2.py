import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from helper_functions import get_appliances
from classes import Household, Neighborhood

def generate_pricing_data():
    pricing = np.zeros(24)
    pricing[0:17] = np.random.uniform(0.45, 0.65, 17)
    pricing[17:20] = np.random.uniform(0.75, 1.0, 3)
    pricing[20:24] = np.random.uniform(0.45, 0.65, 4)
    np.save('data/rt_pricing.npy', pricing)

pricing = np.load('data/rt_pricing.npy')

lonely_house = Household("lonely house")
print(get_appliances(filter_shiftable=0))

lonely_house.add_appliances(get_appliances(filter_shiftable=0)) 
lonely_house.add_appliances(get_appliances(filter_shiftable=1))
#app_to_be_saved = get_random()
print(lonely_house)
#lonely_house.add_appliances(get_random_optional_shiftable())
random_neighborhood = Neighborhood("another lonely", [lonely_house], pricing="RTP")
c, u, l, A_eq, b_eq, A_ub, b_ub = random_neighborhood.get_linprog_input()
#print(A_eq, b_eq)
random_neighborhood.optimize()

#print(random_neighborhood.get_schedule().reshape(-1,24))
