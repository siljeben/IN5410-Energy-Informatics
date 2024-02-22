import pandas as pd


# timeslots for pricing
pricing = ((range(0, 17), 0.5), (range(17, 20), 1), (range(20, 24), 0.5))

expanded_pricing = []
for r, p in pricing:
    for _ in r:  # Use _ as a throwaway variable to indicate that the hour is not being saved
        expanded_pricing.append(p)

df = pd.DataFrame(expanded_pricing, columns=['price'])
print(df)


# data = pd.read_csv("datafile.csv")




