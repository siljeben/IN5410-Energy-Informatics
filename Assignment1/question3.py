import pandas as pd
from question2 import RTP_pricing

from classes import Neighborhood


pricing = RTP_pricing()

data = pd.read_csv("datafile.csv")

neighborhood = Neighborhood("Neighborhood 1", 30, "RTP", 1)

# TODO: Compute the best strategy...



