import numpy as np
from typing import List
from scipy.optimize import linprog
import random
from household import Household
from helper_functions import get_appliances, get_random_optional_shiftable, get_pricing


class Neighborhood():
    houses: List[Household] = []
    house_schedules: List[np.ndarray] = []
    optimized: bool = False
    num_EV: int = 0
    n_households: int = 0
    schedule: np.ndarray 
    pricing: np.ndarray 
    peak_load: float 

    def __init__(self, name: str, households: int | List[Household] = 0, pricing:str = "RTP", peak_load: float = 0) -> None:
        """Function to create a Neighborhood

        Args:
            name (str): The name of a neighborhood
            households (int | List[Household], optional): a list of or number of households in the neighborhood. Defaults to 0.
            pricing (str, optional): The pricing scheme used. Defaults to "RTP".
            peak_load (float, optional): _description_. Defaults to 0.

        Raises:
            ValueError: If the household variable isn't conforming with the constraints
        """
        self.name: str = name
        self.pricing = get_pricing(pricing)
        self.peak_load = peak_load

        if type(households) is int:
            if households < 0:
                raise ValueError("Number of households must be positive.")
            elif households > 0: 
                self.add_random_households(households)
        elif all(isinstance(x, Household) for x in households):
            self.add_households(households)
        else: 
            raise ValueError(f"'households' must be of type int or List[Households], but is of type {type(households)}")   
    
    def add_households(self, households: List[Household]) -> None:
        """Function to add households to a neigborhood

        Args:
            households (List[Household]): A list of households
        """
        self.houses.extend(households)
        self.n_households += len(households)
        self.optimized = False
    
    def add_random_households(self, num_households: int) -> None:
        """A function to add a random number of random households

        Args:
            num_households (int): The number of random households to add
        """
        nonshiftable_appliances = get_appliances(filter_shiftable=0)
        shiftable_appliances: dict = get_appliances(filter_shiftable=1, output_dict=True)
        
        # Removes the EV from shiftable appliances so that it can be used to 
        ev = shiftable_appliances["EV"]        
        shiftable_appliances.pop("EV")

        for i in range(num_households): 
            new_house = Household(f"House {i}")
            
            new_house.add_appliances(nonshiftable_appliances)
            new_house.add_appliances(shiftable_appliances.values())
            
            if random.random() < 0.2: # 20% chance to get an EV at a house
                new_house.add_appliances([ev])

            optional_appliances = get_random_optional_shiftable()
            new_house.add_appliances(optional_appliances)
            
            self.add_households([new_house])
             
    
    def get_linprog_input(self):
        """Function to get the input for linprog

        Returns:
            _type_: _description_
        """
        c = np.array([])
        l = []
        u = []
        A_eq = [[]] #matrix
        b_eq = []
        A_ub = None
        b_ub = None 
        appliance_counter = 0 
        
        for house in self.houses: 
            for appliance in house.appliances: 
                c = np.concatenate((c, self.pricing))
                l = np.concatenate((l, [0 for _ in range(24)]))
                u_temp = np.zeros(24)
                A_eq_temp = np.zeros(24)
                for i in range(appliance.alpha, appliance.beta):
                    u_temp[i] = appliance.hourly_max 
                    A_eq_temp[i] = 1

                if appliance_counter == 0:
                    A_eq = [A_eq_temp]
                else:
                    A_eq = np.append(A_eq, [[0 for _ in range(24)] for _ in range(appliance_counter)], axis=1)
                    A_eq = np.append(A_eq, [np.append([0 for _ in range(24*(appliance_counter))], A_eq_temp)], axis=0)

                u = np.concatenate((u, u_temp))
                b_eq = np.append(b_eq, [appliance.daily_usage_kWh])

                appliance_counter += 1

                if self.peak_load != 0:
                    if A_ub is None: 
                        A_ub = np.identity(24)
                    else: 
                        A_ub = np.append(A_ub, np.identity(24), 1) 
                else:
                    continue
        if A_ub is not None: 
            b_ub = [self.peak_load for _ in range(24)]
        return c, u, l, A_eq, b_eq, A_ub, b_ub

    def optimize(self):
        """Function to optimize the schedule using linprog

        Returns:
            _type_: The result of the optimization
        """
        # optimize, use linprog
        c, u, l, A_eq, b_eq, A_ub, b_ub = self.get_linprog_input()
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, [x for x in zip(l,u)])
        self.optimized = True
        self.schedule = res.x
        return res

    def get_schedule(self):
        """Function that returns the schedule of the neighborhood

        Returns:
            _type_: _description_
        """
        if self.optimized is False: 
            self.optimize()
        return self.schedule
    
    def calc_house_schedules(self):
        """Function to propagate the schedule to the households
        """

        previous_index = 0
        for house in self.houses:
            n_appliances = len(house.appliances)
            self.house_schedules.append(self.schedule[previous_index:previous_index+24*n_appliances].reshape(-1, 24))
            previous_index += n_appliances

    def get_house_schedules(self):
        """Function to get the schedules of the households

        Returns:
            _type_: _description_
        """
        if len(self.house_schedules) == 0:
            self.calc_house_schedules()
        return self.house_schedules
