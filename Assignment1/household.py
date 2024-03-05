from appliance import Appliance
from typing import List
import pickle

class Household():

    def __init__(self, name: str) -> None:
        """Creation of a household object

        Args:
            name (str): Name of the house
        """
        self.name: str = name
        self.appliances: List[Appliance] = []
        self.n_appliances: int = 0
    
    def add_appliances(self, appliances: List[Appliance]) -> None:
        """Function to add a list of appliances to the household

        Args:
            appliances (List[Appliance]): List of appliances
        """
        self.appliances.extend(appliances)
        self.n_appliances += len(appliances)

    def save(self, path: str):
        """Function to save the household to a file

        Args:
            path (str): The path to save the file
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Function to load a household from a file

        Args:
            path (str): The path to load the file from

        Returns:
            Household: The loaded household
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self):
        return f"'{self.name}'(#appliances:{self.n_appliances})"