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


def speed_power_plot(X_train, y_train, model, x_max=15):
    plt.scatter(X_train, y_train, facecolors='none', edgecolors='b', label="Data")
    x = np.linspace(0, x_max, 100)
    pred = model(x)
    plt.plot(x, pred, "r", label="prediction")
    plt.xlabel("Wind speed [m/s]")
    plt.ylabel("Power output [normalized]")
    plt.title("Train data and model")
    plt.show()

