from matplotlib import pyplot as plt
import numpy as np
from typing import List, Optional

def plot_timeseries(time: np.ndarray, data_arrays: List[np.ndarray], labels: List[str], title: str, ylabel: str):
    plt.figure(figsize=(15, 7))
    
    for i, (data, label) in enumerate(zip(data_arrays, labels)):
        if len(data_arrays) == 2:
            plt.plot(data, label=label, color='r' if i==0 else 'b')
    
    plt.legend()
    plt.title(title)
    plt.ylabel(ylabel)

    tick_gap = int(len(time) / 15)
    time_ticks = time[::tick_gap]
    plt.xticks(np.linspace(0, max([len(da) for da in data_arrays]), len(time_ticks)), time_ticks, rotation=45)
    plt.show()

def speed_power_plot(data_x, data_y, model):
    plt.plot(data_x, data_y, 'ro', label="Data")
    x = np.linspace(np.min(data_x), np.max(data_x), 100)
    pred = model(x)
    plt.plot(x, pred, 'b', label='prediction')
    plt.xlabel('Wind speed')
    plt.ylabel('Power output')
    plt.title('Test data over the wind speed')
    plt.show()