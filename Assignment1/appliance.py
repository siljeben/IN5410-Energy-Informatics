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
    def __repr__(self) -> str:
        return f'{self.name} ({self.shiftable}, {self.usage_h}, {self.daily_usage_kWh}, {self.alpha}, {self.beta})'