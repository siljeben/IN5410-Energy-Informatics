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