from typing import List
import numpy as np
from classes_helper_functions import get_pricing
from array import array

class Appliance():
    def __init__(self, name: str, shiftable: int, usage_h: int, daily_usage_kWh: float,   alpha: int, beta: int) -> None:
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
        if shiftable != 0 and beta - alpha == usage_h: # change to enum
            raise ValueError(f"Appliance '{name}' should be shiftable ")
        
    def __repr__(self) -> str:
        return f'{self.name} ({self.shiftable}, {self.usage_h}, {self.daily_usage_kWh}, {self.alpha}, {self.beta})'

class Household():
    appliances: List[Appliance] = []
    n_appliances: int = 0

    def __init__(self, name: str) -> None:
        self.name: str = name
    
    def add_appliances(self, appliances: List[Appliance]) -> None:
        self.appliances.extend(appliances)
        self.n_appliances += len(appliances)
    
    def __repr__(self):
        return f"'{self.name}'({self.n_appliances})"


class Neighborhood():
    houses: List[Household] = []
    optimized: bool = False
    num_EV: int = 0
    n_households: int = 0
    schedule: np.ndarray 
    pricing: np.ndarray 

    def __init__(self, name: str, households: int | List[Household] = 0, pricing:str = "RTP") -> None:
        self.name: str = name
        pricing = get_pricing(pricing)

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
        self.houses.extend(households)
        self.n_households += len(households)
        self.optimized = False
    
    def add_random_households(self, num_households) -> None:
        for i in range(num_households): 
            pass 

    def optimize(self):
        # optimize, use linprog
        c, u, l, A_eq, b_eq, A_ub, b_ub = get_linprog_input(self.houses)
        self.optimized = True
        pass 

    def get_schedule(self):
        if self.optimized is False: 
            self.optimize()
        return self.schedule
