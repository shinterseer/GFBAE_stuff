import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # format datetime axis
import copy
import pandas as pd


class ShoeBox:
    def __init__(self, lengths, heat_max=6000, delta_temperature_max=40, therm_sto=3.6e6, temp_init=21,
                 convective_portion=0.3, u_value=0.25, temperature_supply_delta_max=0.05):
        self.temperature_supply = temp_init
        # self.temperature_supply_delta_max = temperature_supply_delta_max
        self.volume = lengths[0] * lengths[1] * lengths[2]
        self.area_floor = lengths[0] * lengths[1]
        self.area_ceiling = lengths[0] * lengths[1]
        self.area_hull = 2 * lengths[0] * lengths[2] + 2 * lengths[1] * lengths[2]
        air_density = 1.2041  # kg/m3
        air_spec_heat = 1005  # J/K
        self.capacity_air = self.volume * air_spec_heat * air_density  # in J/K
        self.capacity_storage = therm_sto  # in J/K
        self.heat_max = heat_max  # in W
        self.delta_temperature_max = delta_temperature_max
        self.temperature_storage = temp_init
        self.temperature_air = temp_init
        self.convective_portion = convective_portion
        self.u_value = u_value
        self.thermal_transmittance = self.u_value * (self.area_hull + self.area_floor + self.area_ceiling)

    def get_surface_temperature(self):
        return ((self.area_hull + self.area_ceiling) * self.temperature_storage + self.area_floor * self.temperature_supply) / (self.area_hull + self.area_floor + self.area_ceiling)

    def get_operative_temperature(self):
        return 0.5 * (self.temperature_air + self.get_surface_temperature())
        # temp_surf = ((self.area_hull + self.area_ceiling) * self.temperature_storage + self.area_floor * self.temperature_supply) / (self.area_hull + self.area_floor + self.area_ceiling)
        # return 0.5 * (self.temperature_air + temp_surf)

    def timestep(self, heating_power, temperature_outside, time_delta=120):
        # change temperatures
        # heating_power = (self.temperature_supply - self.get_operative_temperature()) / self.delta_temperature_max * self.heat_max
        heating_to_storage = heating_power * (1 - self.convective_portion)
        heating_to_air = heating_power * self.convective_portion
        storage_to_outside = self.thermal_transmittance * (self.temperature_air - temperature_outside)
        Rsi = 0.13  # in m2K/W
        air_to_storage = (self.temperature_air - self.temperature_storage) * (self.area_hull + self.area_ceiling) / Rsi
        dT_air = (heating_to_air - air_to_storage) / self.capacity_air * time_delta
        dT_storage = (heating_to_storage + air_to_storage - storage_to_outside) / self.capacity_storage * time_delta
        self.temperature_air += dT_air
        self.temperature_storage += dT_storage


def model(shoebox, heating_strategy, temperature_outside_series, time_delta):
    """
    this now has to compute the shoebox the whole day
    should return the history of the operative temperature
    """
    # init temperature_operative_series

    num_timesteps = heating_strategy.size
    temperature_operative_series = np.empty(num_timesteps)

    # loop over timesteps
    for i in range(num_timesteps):
        shoebox.timestep(heating_strategy[i], temperature_outside_series[i], time_delta=time_delta)
        # register operative temperature
        temperature_operative_series[i] = shoebox.get_operative_temperature()

    # return temperature_operative_series
    return temperature_operative_series


def cost_function_temperature_setpoint(T_history, T_setpoint):
    """this should now call model to get the history of the operative temperature"""
    # return sum([(T - T_setpoint) ** 2 for T in T_history[1:]])
    return np.sum((T_history[1:] - T_setpoint) ** 2)


def cost_function_demand_response_single(heating_power, power_weight,
                                         temperature_operative, temperature_max, temperature_min,
                                         power_penalty_weight=1., comfort_penalty_weight=1.e9):
    power_penalty = heating_power * power_weight
    comfort_penalty = (max(temperature_operative - temperature_max, 0)
                       + max(temperature_min - temperature_operative, 0))
    # return power_penalty_weight * power_penalty + comfort_penalty_weight * comfort_penalty
    return (power_penalty_weight * power_penalty, comfort_penalty_weight * comfort_penalty)


def cost_function_demand_response2(heating_strategy, power_weight_curve,
                                   temperature_operative_series, temperature_max, temperature_min,
                                   power_penalty_weight=1., comfort_penalty_weight=1.e9):
    # return sum([cost_function_demand_response_single(heating_strategy[i], power_weight_curve[i],
    #                                                  temperature_operative_series[i], temperature_max, temperature_min)
    #            for i in range(heating_strategy.size)])
    mysum = 0
    power_weight_curve_array = power_weight_curve.array
    for i in range(heating_strategy.size):
        cost_temp = cost_function_demand_response_single(heating_strategy[i], power_weight_curve_array[i],
                                                         temperature_operative_series[i], temperature_max, temperature_min,
                                                         power_penalty_weight, comfort_penalty_weight)
        mysum += sum(cost_temp)
    return mysum


def cost_function_demand_response(heating_strategy, power_weight_curve,
                                  temperature_operative_series, temperature_max, temperature_min,
                                  power_penalty_weight=1., comfort_penalty_weight=1.e9):
    # return sum([cost_function_demand_response_single(heating_strategy[i], power_weight_curve[i],
    #                                                  temperature_operative_series[i], temperature_max, temperature_min)
    #            for i in range(heating_strategy.size)])
    # mysum = 0
    # for i in range(heating_strategy.size):
    #     mysum += cost_function_demand_response_single(heating_strategy[i], power_weight_curve[i],
    #                               temperature_operative_series[i], temperature_max, temperature_min)
    # return mysum

    power_penalty = heating_strategy.dot(power_weight_curve)
    comfort_penalty = (np.maximum(temperature_operative_series - temperature_max, 0).sum()
                       + np.maximum(temperature_min - temperature_operative_series, 0).sum())
    # return power_penalty + 1.e9 * comfort_penalty
    return power_penalty_weight * power_penalty + comfort_penalty_weight * comfort_penalty


def cost_wrapper(heating_strategy, shoebox, temperature_outside_series, delta_time, temperature_setpoint,
                 power_weight_curve, temperature_min, temperature_max):
    shoebox_copy = copy.deepcopy(shoebox)
    temperature_operative_series = model(shoebox_copy, heating_strategy, temperature_outside_series, delta_time)
    # return cost_function_temperature_setpoint(temperature_operative_series, temperature_setpoint)
    return cost_function_demand_response(heating_strategy, power_weight_curve,
                                         temperature_operative_series, temperature_max, temperature_min)


def get_load_curve(filename="Lastprofile VDEW_alle.csv", key="Haushalt_Winter"):
    df = pd.read_csv("Lastprofile VDEW_alle.csv")
    df.index = pd.to_datetime(df["Uhrzeit"], format="%H:%M")
    return df[key]


def get_consumption_weight_curve(filename="Lastprofile VDEW_alle.csv", key="Haushalt_Winter"):
    load_curve = get_load_curve(filename, key)
    peak = load_curve.max()
    return load_curve / peak


def get_property_series(shoebox, actuation_strategy, temperature_outside, time_delta, power_weight_curve,
                        temperature_max, temperature_min):
    dim = len(actuation_strategy)
    temperature_storage = np.empty(dim + 1)
    temperature_air = np.empty(dim + 1)
    temperature_operative = np.empty(dim + 1)
    temperature_supply = np.empty(dim + 1)
    temperature_surface = np.empty(dim + 1)
    cost_power = np.empty(dim + 1)
    cost_comfort = np.empty(dim + 1)

    temperature_storage[0] = shoebox.temperature_storage
    temperature_air[0] = shoebox.temperature_air
    temperature_operative[0] = shoebox.get_operative_temperature()
    temperature_supply[0] = shoebox.temperature_air
    temperature_surface[0] = shoebox.get_surface_temperature()
    cost_power[0] = 0
    cost_comfort[0] = 0

    for k in range(dim):
        u = actuation_strategy[k]
        te = temperature_outside[k]
        # shoebox.timestep(te, u)
        shoebox.timestep(u, te, time_delta)
        temperature_storage[k + 1] = shoebox.temperature_storage
        temperature_air[k + 1] = shoebox.temperature_air
        temperature_operative[k + 1] = shoebox.get_operative_temperature()
        temperature_supply[k + 1] = shoebox.temperature_supply
        temperature_surface[k + 1] = shoebox.get_surface_temperature()
        cost_power[k + 1], cost_comfort[k + 1] = cost_function_demand_response_single(u, power_weight_curve.iloc[k],
                                                                                      temperature_operative[k + 1],
                                                                                      temperature_max, temperature_min)
    return {"temperature_air": temperature_air,
            "temperature_storage": temperature_storage,
            "temperature_operative": temperature_operative,
            "temperature_supply": temperature_supply,
            "temperature_surface": temperature_surface,
            "cost_power": cost_power,
            "cost_comfort": cost_comfort}


def array_to_time_series(array, step_in_minutes=5, start_time="2025-04-29 00:00"):
    array = np.array(array)
    step = pd.Timedelta(minutes=step_in_minutes)
    index = pd.date_range(start=pd.Timestamp(start_time), periods=len(array), freq=step)
    return pd.Series(np.array(array), index=index)


def post_proc(actuation_strategy, num_timesteps, T_setpoint,
              lengths, temperature_outside, time_delta, power_weight_curve,
              temperature_min, temperature_max):
    shoebox = ShoeBox(lengths)
    property_dict = get_property_series(shoebox, actuation_strategy=actuation_strategy,
                                        temperature_outside=temperature_outside, time_delta=time_delta,
                                        power_weight_curve=power_weight_curve,
                                        temperature_min=temperature_min, temperature_max=temperature_max)
    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax_top = axes[0, 0]
    # Temperature plot
    # time_steps = np.arange(num_timesteps + 1)
    ax_top.set_ylabel('Temperature in °C', color='b')
    ax_top.tick_params(axis='y', labelcolor='b')
    # ax_top.axhline(y=T_setpoint, color='r', linestyle='--', label='Setpoint')
    # ax1.step(np.arange(num_timesteps), actuation_strategy, 'g-', where='mid', label='Control Actions')
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax1.plot(time_steps, property_dict["temperature_operative"], 'b-', label='Temperature_operative')
    ax_top.plot(array_to_time_series(property_dict["temperature_operative"]), 'b-', label='Temperature_operative')

    # Control actions plot
    ax_top2 = ax_top.twinx()
    # control_steps = np.arange(num_timesteps)
    # ax2.step(control_steps, actuation_strategy, 'g-', where='mid', label='Control Actions')
    # ax_top2.step(array_to_time_series(actuation_strategy), 'g-', where='mid', label='Control Actions')
    ax_top2.plot(array_to_time_series(actuation_strategy), 'g-', label='Control Actions')
    ax_top2.set_ylabel('actuation strategy (heating power in W)')
    # ax2.tick_params(axis='y')

    # Adding legends
    ax_top.legend()
    ax_top.grid()
    # ax2.legend(loc='upper right')

    ax_temperatures = axes[1, 0]
    ax_temperatures.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_temperatures.plot(array_to_time_series(property_dict["temperature_operative"]), label="T_op")
    ax_temperatures.plot(array_to_time_series(property_dict["temperature_air"]), label="T_air")
    ax_temperatures.plot(array_to_time_series(property_dict["temperature_surface"]), label="T_surf")
    ax_temperatures.plot(array_to_time_series(property_dict["temperature_storage"]), label="T_storage")
    ax_temperatures.grid()
    ax_temperatures.legend()

    ax_cost = axes[0, 1]
    ax_cost.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_cost.plot(array_to_time_series(property_dict["cost_power"]), label="cost power")
    ax_cost.plot(array_to_time_series(property_dict["cost_comfort"]), label="cost comfort")
    ax_cost.plot(array_to_time_series(property_dict["cost_power"] + property_dict["cost_comfort"]), label="cost total")
    ax_cost.grid()
    ax_cost.legend()
    ax_cost2 = ax_cost.twinx()
    ax_cost2.plot(array_to_time_series(power_weight_curve), label="power weight curve")
    print(f'final cost sum: {sum(property_dict["cost_power"] + property_dict["cost_comfort"]):.4f}')

    plt.show(block=True)

    x = 0


def main_script():
    # Model parameters
    lengths = (5, 5, 5)
    shoebox = ShoeBox(lengths=lengths)
    temperature_outside = 7
    simulated_time = 24 * 3600
    timestep_in_minutes = 5
    time_delta = 60 * timestep_in_minutes
    num_timesteps = int(simulated_time / time_delta)
    temperature_outside_series = np.full(num_timesteps, temperature_outside)
    power_weight_curve = get_consumption_weight_curve()
    power_weight_curve = power_weight_curve.resample('5min').mean()
    new_time1 = power_weight_curve.index[-1] + pd.Timedelta(minutes=5)
    new_time2 = power_weight_curve.index[-1] + pd.Timedelta(minutes=10)
    power_weight_curve = power_weight_curve.interpolate()
    power_weight_curve[new_time1] = power_weight_curve.iloc[-1]
    power_weight_curve[new_time2] = power_weight_curve.iloc[-1]

    # heating_strategy = np.full(num_timesteps, 1000)
    # temperature_operative_series = model(shoebox, heating_strategy, temperature_outside, delta_time)
    # plt.plot(temperature_operative_series)
    # plt.grid()
    # plt.show(block=True)

    # Initial conditions
    temperature_setpoint = 23  # Desired room temperature
    temperature_min = 20
    temperature_max = 24
    # Constraints
    bounds = [(0, 6000)] * num_timesteps

    # Initial control actions
    heating_power_initial = np.array([1000] * num_timesteps)

    # Optimize control actions
    start_time = time.time()

    minimize_results = minimize(cost_wrapper, heating_power_initial,  # method="Powell",
                                args=(shoebox, temperature_outside_series, time_delta, temperature_setpoint, power_weight_curve,
                                      temperature_min, temperature_max),
                                bounds=bounds)

    actuation_strategy = minimize_results.x
    # Calculate the temperature history with optimal control actions
    # _, T_history = mpc(optimal_control_actions, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out)

    shoebox_new = ShoeBox(lengths=lengths)
    temperature_outside_array = np.full(num_timesteps, temperature_outside)
    # optimal_control_actions = np.ones(optimal_control_actions.size) * 7
    # result_dict = get_property_series(shoebox=shoebox_new, actuation_strategy=actuation_strategy,
    #                                   temperature_outside=temperature_outside_array)
    # T_history = result_dict["temperature_operative"]
    # _, T_history = mpc(optimal_control_actions, shoebox, T_setpoint, temperature_outside,
    #                    prediction_horizon, control_horizon)

    print("actuation_strategy: ", actuation_strategy)
    # print("Function evaluations:", result.nfev)
    print(f"time taken: {time.time() - start_time:.2f} seconds")

    post_proc(actuation_strategy, num_timesteps, temperature_setpoint, lengths,
              temperature_outside=temperature_outside_array, time_delta=time_delta,
              power_weight_curve=power_weight_curve,
              temperature_min=temperature_min, temperature_max=temperature_max)


if __name__ == "__main__":
    # alternative_main()
    main_script()
