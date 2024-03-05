from appliance import Appliance
from typing import List

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
    
    def __repr__(self):
        return f"'{self.name}'(#appliances:{self.n_appliances})"