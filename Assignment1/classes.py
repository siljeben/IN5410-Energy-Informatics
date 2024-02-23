from typing import List
import numpy as np

def get_pricing(pricing: str) -> np.ndarray:
    if pricing == "ToU":
        pricing = np.zeros(24)
        pricing[0:17] = 0.5
        pricing[17:20] = 1.0
        pricing[20:24] = 0.5
    elif pricing == "RTP":
        pricing = np.load('data/rt_pricing.npy')
    else:
        raise ValueError("Pricing must be either 'ToU' or 'RTP'.")
    return pricing

class Appliance():
    def __init__(self, name: str, shiftable: int, usage_kWh: float, usage_h: int, alpha: int, beta: int) -> None:
        self.name: str = name
        self.shiftable = shiftable # TODO: make enum 
        self.usage_kWh: float = usage_kWh 
        self.usage_h: int = usage_h
        self.alpha: int = alpha
        self.beta: int = beta 
        if beta - alpha < usage_h:
            raise ValueError(f"Appliance is used for {usage_h} hours, but usage window is between {alpha} and {beta}.")

class Household():
    appliances: List[Appliance] = None
    n_appliances: int

    def __init__(self, name: str) -> None:
        self.name: str = name
    
    def add_appliances(self, appliances: List[Appliance]) -> None:
        self.appliances.extend(appliances)
        self.n_appliances += len(appliances)


class Neighborhood():
    houses: List[Household] 
    optimized: bool = False
    num_EV: int = 0
    n_households: int = 0
    schedule: np.ndarray
    pricing: np.ndarray

    def __init__(self, name: str, households: int | List[Household] = 0) -> None:
        self.name: str = name
        pricing = get_pricing(pricing)

        if households is int:
            if households < 0:
                raise ValueError("Number of households must be positive.")
            elif households > 0: 
                self.add_random_households(households)
        elif households is List[Household]:
            self.add_households(households)
        
        
    
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
