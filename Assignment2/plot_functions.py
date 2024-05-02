from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_timeseries(
    time: np.ndarray,
    data_arrays: List[np.ndarray],
    labels: List[str],
    title: str,
    ylabel: str,
    figsize: Tuple[int, int] = [15, 7],
    savepath: str = None,
    **kwargs
):
    plt.figure(figsize=figsize)

    colors = ["r", "b", "g", "c"]

    for i, (data, label) in enumerate(zip(data_arrays, labels)):
        color_index = i % len(colors)
        plt.plot(data, label=label, color=colors[color_index], **kwargs)

    plt.legend()
    plt.title(title)
    plt.ylabel(ylabel)

    tick_gap = int(len(time) / 15)
    time_ticks = time[::tick_gap]
    plt.xticks(
        np.linspace(0, max([len(da) for da in data_arrays]), len(time_ticks)),
        time_ticks,
        rotation=45,
    )
    plt.xlabel("Year 2013")
    if savepath:
        plt.savefig(savepath)
    plt.show()


def speed_power_plot(data_x, data_y, model):
    plt.plot(data_x, data_y, "ro", label="Data")
    x = np.linspace(np.min(data_x), np.max(data_x), 100)
    pred = model(x)
    plt.plot(x, pred, "b", label="prediction")
    plt.xlabel("Wind speed")
    plt.ylabel("Power output")
    plt.title("Test data over the wind speed")
    plt.show()

def speed_power_plot_sklearn(X_train, y_train, model, max=15): 
    plt.scatter(X_train, y_train, facecolors='none', edgecolors='b', label="Data")
    x = np.linspace(0, max, 100)
    pred = model.predict(x.reshape(-1,1))
    plt.plot(x, pred, "r", label="prediction")
    plt.xlabel("Wind speed [m/s]")
    plt.ylabel("Power output [normalized]")
    plt.title("Train data and model")
    plt.show()
    


