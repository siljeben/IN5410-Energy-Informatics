from household import Household
from neighborhood import Neighborhood
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def plot_schedule_appliances(neighborhood: Neighborhood, include_house_name=True):
    fig, ax = plt.subplots()
    hours = np.arange(24)
    bottom = np.zeros(24)

    for (house, schedule) in zip(neighborhood.houses, neighborhood.get_house_schedules()):
        for i in range(house.n_appliances):
            label = house.appliances[i].name
            if include_house_name: 
                label += f", {house.name}"
            plt.bar(hours+0.5, height=schedule[i], bottom=bottom, label=label, color=cm.tab20(i/len(schedule)))
            bottom += schedule[i]
    plt.xticks(np.arange(0, 25, 2))
    plt.xlabel("Time [h]")
    plt.ylabel("Power usage [kWh]")

    ax2 = ax.twinx()
    ax2.set_ylabel('Electricity price [NOK/kWh]')
    ax2.set_yticks(np.arange(0, 1.1, 0.5))
    ax2.set_ylim(0, 1.1)
    price_hours = np.concatenate((np.repeat(np.arange(24), 2)[1::], [24]))
    ax2.plot(price_hours, np.repeat(neighborhood.pricing, 2), '--', color='black', label="Electricity price")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.show()


def plot_schedule_shiftable_nonshiftable(neighborhood: Neighborhood):
    pricing = neighborhood.pricing

    fig, ax = plt.subplots()
    hours = np.arange(24)
    bottom = np.zeros(24)

    shiftable = np.zeros(24)
    nonshiftable = np.zeros(24)
    for (house, schedule) in zip(neighborhood.houses, neighborhood.house_schedules):
        for i in range(schedule.shape[0]):
            appliance = house.appliances[i]
            if appliance.shiftable == 1 or (appliance.shiftable == 2 and appliance.usage_h < appliance.beta - appliance.alpha):
                shiftable += schedule[i]
            else:
                nonshiftable += schedule[i]

    plt.bar(hours+0.5, height=nonshiftable, bottom=bottom, label="Non-shiftable")
    bottom += nonshiftable
    plt.bar(hours+0.5, height=shiftable, bottom=bottom, label="Shiftable")

    plt.xticks(np.arange(0, 25, 2))
    plt.xlabel("Time [h]")
    plt.ylabel("Power usage [kWh]")

    ax2 = ax.twinx()
    ax2.set_ylabel('Electricity price [NOK/kWh]')
    ax2.set_yticks(np.arange(0, 1.1, 0.5))
    ax2.set_ylim(0, 1.1)
    price_hours = np.concatenate((np.repeat(np.arange(24), 2)[1::], [24]))
    ax2.plot(price_hours, np.repeat(pricing, 2), '--', color='black', label="Electricity price")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    pass