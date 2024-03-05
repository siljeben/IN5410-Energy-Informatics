import pandas as pd
from question2 import RTP_pricing

from classes import Neighborhood


pricing = RTP_pricing()

data = pd.read_csv("datafile.csv")

neighborhood = Neighborhood(name="Neighborhood 1", households=30, pricing="RTP", peak_load=0)

# TODO: Compute the best strategy...



