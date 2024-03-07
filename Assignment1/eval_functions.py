from household import Household
from neighborhood import Neighborhood
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def plot_household_schedule_appliances(house: Household, schedule: np.ndarray, pricing: np.ndarray) -> float:
    """Function to plot the schedule of a household and calculate the cost of the schedule

    Args:
        schedule (np.ndarray): The schedule of a household
        pricing (np.ndarray): The pricing of the electricity

    Returns:
        float: The cost of the schedule
    """
    fig, ax = plt.subplots()
    hours = np.arange(24)
    bottom = np.zeros(len(schedule[0]))
    for i in range(schedule.shape[0]):
        plt.bar(hours+0.5, height=schedule[i], bottom=bottom, label=house.appliances[i].name, color=cm.tab20(i/len(schedule)))
        bottom += schedule[i]
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

def plot_household_schedule_shiftable_nonshiftable(house: Household, schedule: np.ndarray, pricing: np.ndarray) -> float:
    """Function to plot the schedule of a household and calculate the cost of the schedule

    Args:
        schedule (np.ndarray): The schedule of a household
        pricing (np.ndarray): The pricing of the electricity

    Returns:
        float: The cost of the schedule
    """
    fig, ax = plt.subplots()
    hours = np.arange(24)
    bottom = np.zeros(len(schedule[0]))

    shiftable = np.zeros(24)
    nonshiftable = np.zeros(24)
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
    random_neighborhood = Neighborhood("another lonely", pricing="RTP")

    # random_neighborhood.add_random_households(1)
    # random_neighborhood.houses[0].save('data/random_household.pkl')

    random_household = Household.load('data/random_household.pkl')
    print(random_household.appliances)
    random_neighborhood.add_households([random_household])

    c, u, l, A_eq, b_eq, A_ub, b_ub = random_neighborhood.get_linprog_input()
    #print(A_eq, b_eq)
    random_neighborhood.optimize()
    house_schedules = random_neighborhood.get_house_schedules()

    plot_household_schedule_appliances(random_neighborhood.houses[0], house_schedules[0], random_neighborhood.pricing)
    plot_household_schedule_shiftable_nonshiftable(random_neighborhood.houses[0], house_schedules[0], random_neighborhood.pricing)
