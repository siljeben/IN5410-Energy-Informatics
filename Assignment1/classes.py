import numpy as np
from typing import List
from array import array
from classes_helper_functions import get_pricing
from scipy.optimize import linprog
import random

class Appliance():
    def __init__(self, name: str, shiftable: int, usage_h: int, daily_usage_kWh: float, alpha: int, beta: int) -> None:
        """Class that models an appliance and its power usage

        Args:
            name (str): The name of the appliance
            shiftable (int): 0 - if it's nonshiftable, 1 if it's shiftable and 2 if it's shiftable, but not required
            usage_h (int): Amount of hours it should be used
            daily_usage_kWh (float): The daily usage of the appliance in KWh
            alpha (int): The start time where the appliance can be used
            beta (int): The end time where the appliance can be used

        Raises:
            ValueError: If the appliance data isn't conforming with the constraints
        """
        self.name: str = name
        self.shiftable = shiftable # TODO: make enum 
        self.usage_h: int = usage_h
        self.daily_usage_kWh: float = daily_usage_kWh 
        self.hourly_max: float = daily_usage_kWh / usage_h
        self.alpha: int = alpha
        self.beta: int = beta 
        if beta - alpha < usage_h:
            raise ValueError(f"Appliance '{name}' is used for {usage_h} hours, but usage window is between {alpha} and {beta}.")
        if shiftable == 0 and beta - alpha > usage_h: #change when enum is made
            raise ValueError(f"Appliance '{name}' is not shiftable. Window between {alpha} and {beta} gives a range of {beta - alpha} hours, but usage should be {usage_h} hours.")
        if shiftable == 1 and beta - alpha == usage_h: # change to enum
            raise ValueError(f"Appliance '{name}' should be shiftable ")
        
    def __str__(self) -> str:
        return f'{self.name} ({self.shiftable}, {self.usage_h}, {self.daily_usage_kWh}, {self.alpha}, {self.beta})'

class Household():
    appliances: List[Appliance] = []
    n_appliances: int = 0

    def __init__(self, name: str) -> None:
        """Creation of a household object

        Args:
            name (str): Name of the house
        """
        self.name: str = name
    
    def add_appliances(self, appliances: List[Appliance]) -> None:
        """Function to add a list of appliances to the household

        Args:
            appliances (List[Appliance]): List of appliances
        """
        self.appliances.extend(appliances)
        self.n_appliances += len(appliances)
    
    def __str__(self):
        return f"'{self.name}'(#appliances:{self.n_appliances})"


class Neighborhood():
    houses: List[Household] = []
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
        nonshiftable_appliances = get_appliances(filter_shiftable=0) # TODO: Actually import the appliances
        shiftable_appliances = get_appliances(filter_shiftable=1) # TODO: Actually import the appliances
        optional_appliances = get_appliances(filter_shiftable=2, random_selection_n=2) # TODO: Actually import the appliances and add the number of appliances we actually want
        
        # Removes the EV from shiftable appliances so that it can be used to 
        # Also assumes that the EV is the last in the list, which might be a wrong assumption
        ev = shiftable_appliances[-1]
        shiftable_appliances.pop(-1)

        for i in range(num_households): 
            new_house = Household(f"House {i}")
            
            new_house.add_appliances(nonshiftable_appliances)
            new_house.add_appliances(shiftable_appliances)
            new_house.add_appliances(optional_appliances)
            
            if random.random() < 0.2: # 20% chance to get an EV at a house
                new_house.add_appliances(ev)
            
            self.add_households(new_house)
             
    
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
