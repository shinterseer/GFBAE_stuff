import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # format datetime axis
import copy
import pandas as pd
import pickle
from functools import partial

from numba import float64
from numba.experimental import jitclass
from numba import jit

import pso

spec = [
    ('temperature_supply', float64),
    ('volume', float64),
    ('area_ceiling', float64),
    ('area_hull', float64),
    ('area_floor', float64),
    ('area_ceiling', float64),
    ('area_hull', float64),
    ('air_density', float64),
    ('air_spec_heat', float64),
    ('capacity_air', float64),
    ('capacity_storage', float64),
    ('heat_max', float64),
    ('delta_temperature_max', float64),
    ('temperature_storage', float64),
    ('temperature_air', float64),
    ('convective_portion', float64),
]


@jitclass(spec)
class ShoeBox:
    def __init__(self, length1=5., length2=5., length3=5., heat_max=6000., delta_temperature_max=40., therm_sto=3.6e6, temp_init=21.,
                 convective_portion=0.3, u_value=0.3, temperature_supply_delta_max=0.05):
        self.temperature_supply = temp_init
        # self.temperature_supply_delta_max = temperature_supply_delta_max
        self.volume = length1 * length2 * length3
        self.area_floor = length1 * length2
        self.area_ceiling = length1 * length2
        self.area_hull = 2 * length1 * length3 + 2 * length2 * length3
        self.air_density = 1.2041  # kg/m3
        self.air_spec_heat = 1005  # J/K
        self.capacity_air = self.volume * self.air_spec_heat * self.air_density  # in J/K
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
        # heating_power = self.heat_max * heating_power
        heating_power = min(self.heat_max, heating_power)
        heating_power = max(0, heating_power)

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


@jit
def model(shoebox, heating_strategy, temperature_outside_series, time_delta, substeps_per_actuation):
    """
    this now has to compute the shoebox the whole day
    should return the history of the operative temperature
    """
    # init temperature_operative_series

    num_actuation_steps = heating_strategy.size
    temperature_operative_series = np.empty(num_actuation_steps * substeps_per_actuation)
    temperature_surface_series = np.empty(num_actuation_steps * substeps_per_actuation)
    temperature_air_series = np.empty(num_actuation_steps * substeps_per_actuation)
    temperature_storage_series = np.empty(num_actuation_steps * substeps_per_actuation)

    # loop over timesteps
    for i in range(num_actuation_steps):
        for j in range(substeps_per_actuation):
            shoebox.timestep(heating_strategy[i], temperature_outside_series[i], time_delta=time_delta)
            # register operative temperature
            temperature_operative_series[i * substeps_per_actuation + j] = shoebox.get_operative_temperature()
            temperature_surface_series[i * substeps_per_actuation + j] = shoebox.get_surface_temperature()
            temperature_air_series[i * substeps_per_actuation + j] = shoebox.temperature_air
            temperature_storage_series[i * substeps_per_actuation + j] = shoebox.temperature_storage

    result_dict = {'temperature_operative_series': temperature_operative_series,
                   'temperature_surface_series': temperature_surface_series,
                   'temperature_air_series': temperature_air_series,
                   'temperature_storage_series': temperature_storage_series}

    return result_dict


def cost_function_temperature_setpoint(T_history, T_setpoint):
    """this should now call model to get the history of the operative temperature"""
    # return sum([(T - T_setpoint) ** 2 for T in T_history[1:]])
    return np.sum((T_history[1:] - T_setpoint) ** 2)


def cost_function_demand_response3(heating_strategy, power_weight_curve,
                                   temperature_operative_series, temperature_max, temperature_min,
                                   power_penalty_weight=1., comfort_penalty_weight=1.e5):
    power_penalty = np.dot(heating_strategy, power_weight_curve)

    temperature_mid = 0.5 * (temperature_max + temperature_min)
    # temperature_mid_array = np.full(heating_strategy.size, 0.5 * (temperature_max + temperature_min))
    comfort_penalty_array = (temperature_operative_series - temperature_mid) / (temperature_max - temperature_mid)
    comfort_penalty_array2 = comfort_penalty_array * comfort_penalty_array
    comfort_penalty_array4 = comfort_penalty_array2 * comfort_penalty_array2
    comfort_penalty_array8 = comfort_penalty_array4 * comfort_penalty_array4
    comfort_penalty_array16 = comfort_penalty_array8 * comfort_penalty_array8
    comfort_penalty_array32 = comfort_penalty_array16 * comfort_penalty_array16
    comfort_penalty_array64 = comfort_penalty_array32 * comfort_penalty_array32
    comfort_penalty = comfort_penalty_array64.sum()

    # comfort_penalty = np.power((temperature_operative_series - temperature_mid_array), 32).sum()
    # comfort_penalty = sum([(temperature_operative - temperature_mid)**32 for temperature_operative in temperature_operative_series])
    return power_penalty_weight * power_penalty + comfort_penalty_weight * comfort_penalty


def cost_function_demand_response(heating_strategy, power_weight_curve,
                                  temperature_operative_series, temperature_max, temperature_min, heating_min=0, heating_max=6000,
                                  power_penalty_weight=1., comfort_penalty_weight=1.e5, control_penalty_weight=1.e6):
    control_penalty_array = (np.maximum(heating_strategy - heating_max, 0)
                             + np.maximum(heating_min - heating_strategy, 0))

    heating_strategy = np.maximum(heating_strategy, heating_min)
    heating_strategy = np.minimum(heating_strategy, heating_max)

    power_penalty_array = heating_strategy * power_weight_curve
    comfort_penalty_base_array = (temperature_operative_series - 0.5 * (temperature_max + temperature_min)) / comfort_penalty_weight
    comfort_penalty_soft_array = (np.maximum(temperature_operative_series - temperature_max - .1, 0)
                                  + np.maximum(temperature_min + .1 - temperature_operative_series, 0)) / np.sqrt(comfort_penalty_weight)
    comfort_penalty_hard_array = (np.maximum(temperature_operative_series - temperature_max, 0)
                                  + np.maximum(temperature_min - temperature_operative_series, 0))
    # comfort_penalty_array = comfort_penalty_base_array + comfort_penalty_soft_array + comfort_penalty_hard_array
    comfort_penalty_array = comfort_penalty_hard_array

    if max(comfort_penalty_hard_array) > 0:
        x = 0

    return {"cost_total": (power_penalty_weight * power_penalty_array.sum()
                           + comfort_penalty_weight * comfort_penalty_array.sum()
                           + control_penalty_weight * control_penalty_array.sum()),
            "cost_power": power_penalty_weight * power_penalty_array,
            "cost_comfort": comfort_penalty_weight * comfort_penalty_array,
            "cost_control": control_penalty_weight * control_penalty_array}


def cost_wrapper(heating_strategy, shoebox, temperature_outside_series, delta_time,
                 power_weight_curve, temperature_min, temperature_max, substeps_per_actuation,
                 comfort_penalty_weight=1.e5, control_penalty_weight=1.e6, return_full_dict=False):
    shoebox_copy = copy.deepcopy(shoebox)
    result_dict = model(shoebox_copy, heating_strategy, temperature_outside_series, delta_time, substeps_per_actuation)
    temperature_operative_series = result_dict['temperature_operative_series']

    cost_dict = cost_function_demand_response(np.repeat(heating_strategy, substeps_per_actuation), power_weight_curve,
                                              temperature_operative_series, temperature_max, temperature_min,
                                              comfort_penalty_weight=comfort_penalty_weight, control_penalty_weight=control_penalty_weight)
    # return np.dot(cost_dict['cost_comfort'], cost_dict['cost_comfort']) #+ np.dot(cost_dict['cost_control'], cost_dict['cost_control'])
    if return_full_dict:
        return cost_dict
    return cost_dict['cost_total']


def get_load_curve(filename="Lastprofile VDEW_alle.csv", key="Haushalt_Winter"):
    df = pd.read_csv("Lastprofile VDEW_alle.csv")
    df.index = pd.to_datetime(df["Uhrzeit"], format="%H:%M")
    return df[key]


def get_consumption_weight_curve(resample_in_minutes, filename="Lastprofile VDEW_alle.csv", key="Haushalt_Winter"):
    load_curve = get_load_curve(filename, key)
    new_time1 = load_curve.index[-1] + pd.Timedelta(minutes=15)
    load_curve[new_time1] = load_curve.iloc[-1]
    load_curve = load_curve.resample(f'{resample_in_minutes}min').mean()
    load_curve = load_curve.interpolate()
    load_curve = load_curve[:-1]  # kick last one
    peak = load_curve.max()
    return load_curve / peak


def get_postproc_info(shoebox, actuation_sequence, temperature_outside, time_delta, power_weight_curve,
                      temperature_max, temperature_min, substeps_per_actuation, comfort_penalty_weight=1.e5):
    # actuation_sequence = np.repeat(actuation_sequence, substeps_per_actuation)
    dim = len(actuation_sequence)
    total_energy_turnover = time_delta * np.sum(np.abs(actuation_sequence))
    grid_burden = time_delta * np.dot(np.repeat(actuation_sequence, substeps_per_actuation), power_weight_curve)
    peak_alignment_factor = grid_burden / total_energy_turnover

    temperature_storage = np.empty(dim)
    temperature_air = np.empty(dim)
    temperature_operative = np.empty(dim)
    temperature_supply = np.empty(dim)
    temperature_surface = np.empty(dim)
    # cost_power = np.empty(dim + 1)
    # cost_comfort = np.empty(dim + 1)

    # temperature_storage[0] = shoebox.temperature_storage
    # temperature_air[0] = shoebox.temperature_air
    # temperature_operative[0] = shoebox.get_operative_temperature()
    # temperature_supply[0] = shoebox.temperature_air
    # temperature_surface[0] = shoebox.get_surface_temperature()
    # cost_power[0] = 0
    # cost_comfort[0] = 0

    result_dict = model(shoebox, heating_strategy=actuation_sequence, temperature_outside_series=temperature_outside,
                        time_delta=time_delta, substeps_per_actuation=substeps_per_actuation)

    # for k in range(dim):
    #     u = actuation_sequence[k]
    #     te = temperature_outside[k]
    #     shoebox.timestep(u, te, time_delta)
    #     temperature_storage[k + 1] = shoebox.temperature_storage
    #     temperature_air[k + 1] = shoebox.temperature_air
    #     temperature_operative[k + 1] = shoebox.get_operative_temperature()
    #     temperature_supply[k + 1] = shoebox.temperature_supply
    #     temperature_surface[k + 1] = shoebox.get_surface_temperature()
    # cost_power[k + 1], cost_comfort[k + 1] = cost_function_demand_response_single(u, power_weight_curve.iloc[k],
    #                                                                               temperature_operative[k + 1],
    #                                                                               temperature_max, temperature_min,
    #                                                                               comfort_penalty_weight=comfort_penalty_weight)

    cost_dict = cost_function_demand_response(np.repeat(actuation_sequence, substeps_per_actuation), power_weight_curve,
                                              temperature_operative, temperature_max, temperature_min,
                                              comfort_penalty_weight=comfort_penalty_weight)

    return {"temperature_air": result_dict['temperature_air_series'],
            "temperature_storage": result_dict['temperature_storage_series'],
            "temperature_operative": result_dict['temperature_operative_series'],
            # "temperature_supply": temperature_supply,
            "temperature_surface": result_dict['temperature_surface_series'],
            "cost_power": cost_dict['cost_power'],
            "cost_comfort": cost_dict['cost_comfort'],
            "cost_control": cost_dict['cost_control'],
            "total_energy_turnover": total_energy_turnover,
            "grid_burden": grid_burden,
            "peak_alignment_factor": peak_alignment_factor}


def array_to_time_series(array, step_in_minutes=1, start_time="2025-04-29 00:00"):
    array = np.array(array)
    step = pd.Timedelta(minutes=step_in_minutes)
    index = pd.date_range(start=pd.Timestamp(start_time), periods=len(array), freq=step)
    return pd.Series(np.array(array), index=index)


def post_proc(postproc_dict, actuation_sequence, power_weight_curve):
    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax_top = axes[0, 0]
    # Temperature plot
    # time_steps = np.arange(num_timesteps + 1)
    ax_top.set_ylabel('Temperature in °C', color='b')
    ax_top.tick_params(axis='y', labelcolor='b')
    # ax_top.axhline(y=T_setpoint, color='r', linestyle='--', label='Setpoint')
    # ax1.step(np.arange(num_timesteps), actuation_sequence, 'g-', where='mid', label='Control Actions')
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax1.plot(time_steps, postproc_dict["temperature_operative"], 'b-', label='Temperature_operative')
    ax_top.plot(array_to_time_series(postproc_dict["temperature_operative"]), 'b-', label='Temperature_operative')

    # Control actions plot
    ax_top2 = ax_top.twinx()
    # control_steps = np.arange(num_timesteps)
    # ax2.step(control_steps, actuation_sequence, 'g-', where='mid', label='Control Actions')
    # ax_top2.step(array_to_time_series(actuation_sequence), 'g-', where='mid', label='Control Actions')
    ax_top2.plot(array_to_time_series(actuation_sequence), 'g-', label='Control Actions')
    ax_top2.set_ylabel('actuation sequence (heating power in W)')
    # ax2.tick_params(axis='y')

    # Adding legends
    ax_top.legend()
    ax_top.grid()
    # ax2.legend(loc='upper right')

    ax_temperatures = axes[1, 0]
    ax_temperatures.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_temperatures.plot(array_to_time_series(postproc_dict["temperature_operative"]), label="T_op")
    ax_temperatures.plot(array_to_time_series(postproc_dict["temperature_air"]), label="T_air")
    ax_temperatures.plot(array_to_time_series(postproc_dict["temperature_surface"]), label="T_surf")
    ax_temperatures.plot(array_to_time_series(postproc_dict["temperature_storage"]), label="T_storage")
    ax_temperatures.grid()
    ax_temperatures.legend()

    ax_cost = axes[0, 1]
    ax_cost.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_cost.plot(array_to_time_series(postproc_dict["cost_power"]), label="cost power", color="orange")
    ax_cost.grid()
    ax_cost.legend()
    ax_cost2 = ax_cost.twinx()
    ax_cost2.plot(array_to_time_series(power_weight_curve), label="power weight curve")

    ax_cost_comfort = axes[1, 1]
    ax_cost_comfort.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_cost_comfort.plot(array_to_time_series(postproc_dict["cost_comfort"]), label="cost comfort", color='orange')
    ax_cost_comfort.grid()
    ax_cost_comfort.legend()
    ax_cost_comfort2 = ax_cost_comfort.twinx()
    ax_cost_comfort2.plot(array_to_time_series(postproc_dict["temperature_operative"]), label="T_op")

    plt.tight_layout()
    plt.show(block=True)


def get_basic_parameters():
    temperature_outside = 7
    simulated_time = 24 * 3600
    actuationstep_in_minutes = 60
    substeps_per_actuation = int(actuationstep_in_minutes / 1)  # make one simulationstep every 1 minutes
    simulationstep_in_minutes = -1
    if actuationstep_in_minutes % substeps_per_actuation == 0:
        simulationstep_in_minutes = actuationstep_in_minutes / substeps_per_actuation
    else:
        print("actuationstep_in_minutes % substeps_per_actuation != 0")
        return 1

    time_delta = 60 * simulationstep_in_minutes  # time_delta in seconds = 60 seconds per minute times number of minutes
    num_timesteps = int(simulated_time / time_delta)
    num_actuation_steps = int(simulated_time / (60 * actuationstep_in_minutes))
    temperature_outside_series = np.full(num_timesteps, temperature_outside)

    power_weight_curve = get_consumption_weight_curve(resample_in_minutes=simulationstep_in_minutes)

    # Initial conditions
    temperature_min = 20
    temperature_max = 24
    # Constraints
    bounds = [(0, 6000)] * num_actuation_steps

    # Initial control actions
    heating_power_initial = np.array([500] * num_actuation_steps)

    lengths = np.array((5., 5., 5.))
    return {"lengths": lengths,
            "bounds": bounds,
            "heating_power_initial": heating_power_initial,
            "temperature_outside_series": temperature_outside_series,
            "temperature_min": temperature_min,
            "temperature_max": temperature_max,
            "simulationstep_in_minutes": simulationstep_in_minutes,
            "substeps_per_actuation": substeps_per_actuation,
            "time_delta": time_delta,
            "num_timesteps": num_timesteps,
            "num_actuation_steps": num_actuation_steps,
            "power_weight_curve": power_weight_curve}


def main_script():
    # Model parameters
    basic_parameter_dict = get_basic_parameters()
    temperature_outside_series = basic_parameter_dict["temperature_outside_series"]
    time_delta = basic_parameter_dict["time_delta"]
    power_weight_curve = basic_parameter_dict["power_weight_curve"]
    heating_power_initial = basic_parameter_dict["heating_power_initial"]
    bounds = basic_parameter_dict["bounds"]
    temperature_min = basic_parameter_dict["temperature_min"]
    temperature_max = basic_parameter_dict["temperature_max"]
    substeps_per_actuation = basic_parameter_dict["substeps_per_actuation"]
    lengths = basic_parameter_dict["lengths"]

    # Optimize control actions
    print("u-value, thermal capacity, comfort penalty weight, peak alignment factor, total energy turnover, grid burden, cost power, cost comfort, cost_control, computation time in s")
    # print("thermal capacity, peak alignment factor, total energy turnover, grid burden, computation time in s")
    df_results = pd.DataFrame(columns=["u-value", "thermal capacity", "comfort penaltyweight",
                                       "peak alignment factor", "total energy turnover", "grid burden", "cost power", "cost comfort", 'cost_control',
                                       "computation time"])

    # df_results = pd.DataFrame(columns=["thermal capacity", "peak alignment factor", "total energy turnover", "grid burden", "computation time"])
    # u_values = [0.1, .15, 0.2, .25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # thermal_capacities = 1.e6 * np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.])
    thermal_capacities = 1.e6 * np.array([8.])
    # thermal_capacities = 1.e6 * np.array([1., 2.])
    # u_values = [0.3]

    # comfort_penalty_weights = [1.e5, 1.e6, 1.e7, 1.e8, 1.e9, 1.e10]
    comfort_penalty_weight = 1.e7
    control_penalty_weight = 1.e6
    # therm_sto = 3.6e6
    u_value = .3
    # filename_suffix = f"u{u_value}_tc{therm_sto:.1e}_cpwV"
    # filename_suffix = f"uV_tc{therm_sto:.1e}_cpw{comfort_penalty_weight:.1e}"
    filename_suffix = f"u{u_value}_tcV_cpw{comfort_penalty_weight:.1e}"
    # df_actuation_results = pd.DataFrame(columns=comfort_penalty_weights)
    # df_actuation_results = pd.DataFrame(columns=u_values)
    df_actuation_results = pd.DataFrame(columns=thermal_capacities)

    for i in range(len(thermal_capacities)):
        # for i in range(len(u_values)):
        # for i in range(len(comfort_penalty_weights)):
        #     u_value = u_values[i]
        therm_sto = thermal_capacities[i]
        # comfort_penalty_weight = comfort_penalty_weights[i]

        start_time = time.time()
        # self, length1=lengths, length2=length[1], length3=lengths[2], heat_max=6000, delta_temperature_max=40, therm_sto=therm_sto, temp_init=20,
        # convective_portion=0.3, u_value=u_value, temperature_supply_delta_max=0.05


        # shoebox = ShoeBox(temp_init=20, length1=lengths[0], length2=lengths[1], length3=lengths[2], u_value=u_value, therm_sto=therm_sto)
        # shoebox = ShoeBox(length1=lengths[0], length2=lengths[1], length3=lengths[2], heat_max=6000, delta_temperature_max=40, therm_sto=therm_sto, temp_init=20,
        #     convective_portion=0.3, u_value=u_value, temperature_supply_delta_max=0.05)
        # shoebox = ShoeBox(lengths[0], lengths[1], lengths[2], 6000., 40, therm_sto, 20.,
        #     0.3, u_value, 0.05)

        shoebox = ShoeBox(u_value=float64(u_value), therm_sto=therm_sto)



        shoebox_init = copy.deepcopy(shoebox)

        # def wrapped_func(x):
        #     return cost_wrapper(x, shoebox, temperature_outside_series, time_delta, power_weight_curve,
        #                         temperature_min, temperature_max, substeps_per_actuation, comfort_penalty_weight, control_penalty_weight, return_full_dict=True)

        wrapped_func = partial(cost_wrapper, shoebox=shoebox, temperature_outside_series=temperature_outside_series,
                               delta_time=time_delta, power_weight_curve=power_weight_curve,
                               temperature_min=temperature_min, temperature_max=temperature_max,
                               substeps_per_actuation=substeps_per_actuation, comfort_penalty_weight=comfort_penalty_weight,
                               control_penalty_weight=control_penalty_weight, return_full_dict=True)

        # def cost_wrapper(heating_strategy, shoebox, temperature_outside_series, delta_time,
        #                  power_weight_curve, temperature_min, temperature_max, substeps_per_actuation,
        #                  comfort_penalty_weight=1.e5, control_penalty_weight=1.e6, return_full_dict=False):

        solution = pso.pso(wrapped_func, dim=24, n_particles=30, n_iters=300, print_every=50, bounds=(0, 6000),
                           stepsize=500, c1=1, c2=1, randomness=.5, visualize=False, num_processes=1)
        actuation_sequence = solution[0]

        # print('checking cost from main with wrapped_func - cost_wrapper')
        # pso.check_cost(actuation_sequence, wrapped_func)

        # minimize_results = minimize(cost_wrapper, heating_power_initial,
        #
        #                             # method='BFGS',
        #                             # options={
        #                             #     'gtol': 1e-10,
        #                             #     'eps': 1e-8,
        #                             #     'maxiter': 10000,
        #                             #     'disp': True
        #                             # },
        #
        #                             args=(shoebox, temperature_outside_series, time_delta, power_weight_curve,
        #                                   temperature_min, temperature_max, substeps_per_actuation, comfort_penalty_weight),
        #                             bounds=bounds)

        # minimize_results = basinhopping(wrapped_func, heating_power_initial, niter=20)
        # actuation_sequence = minimize_results.x

        shoebox_fresh = copy.deepcopy(shoebox_init)
        postproc_dict = get_postproc_info(shoebox=shoebox_fresh, actuation_sequence=actuation_sequence,
                                          temperature_outside=temperature_outside_series, time_delta=time_delta,
                                          power_weight_curve=power_weight_curve,
                                          temperature_min=temperature_min, temperature_max=temperature_max,
                                          substeps_per_actuation=substeps_per_actuation,
                                          comfort_penalty_weight=comfort_penalty_weight)
        computation_time = time.time() - start_time

        # df_results = pd.DataFrame(columns=["u-value", "thermal capacity", "comfort penaltyweight",
        #                                    "peak alignment factor", "total energy turnover", "grid burden",
        #                                    "computation time"])

        df_results.loc[i] = [u_value, therm_sto, comfort_penalty_weight,
                             postproc_dict['peak_alignment_factor'],
                             postproc_dict['total_energy_turnover'],
                             postproc_dict['grid_burden'],
                             postproc_dict['cost_power'].sum(),
                             postproc_dict['cost_comfort'].sum(),
                             postproc_dict['cost_control'].sum(),
                             computation_time]

        df_actuation_results.iloc[:, i] = actuation_sequence

        print(f"{u_value}, {therm_sto}, {comfort_penalty_weight:.1e}, "
              f"{postproc_dict['peak_alignment_factor']:.4f}, "
              f"{postproc_dict['total_energy_turnover']:.3e}, "
              f"{postproc_dict['grid_burden']:.3e}, "
              f"{postproc_dict['cost_power'].sum():.3e}, "
              f"{postproc_dict['cost_comfort'].sum():.3e}, "
              f"{postproc_dict['cost_control'].sum():.3e}, "
              f"{computation_time:.3f}")

    df_results.to_csv('shoebox_results.csv')
    df_actuation_results.to_csv(f"actuation_results_{filename_suffix}.csv")
    post_proc(postproc_dict, actuation_sequence=np.repeat(actuation_sequence, substeps_per_actuation), power_weight_curve=power_weight_curve)


def plot_grid_stress_index(shoebox, filename="20250508_actuation_results_u0.3_tc3.6e+06_cpwV.csv", x_label=None):
    df = pd.read_csv(filename, index_col=0)
    temperature_outside_series = get_basic_parameters()["temperature_outside_series"]
    time_delta = get_basic_parameters()["time_delta"]
    power_weight_curve = get_basic_parameters()["power_weight_curve"]
    temperature_min = get_basic_parameters()["temperature_min"]
    temperature_max = get_basic_parameters()["temperature_max"]
    substeps_per_actuation = get_basic_parameters()["substeps_per_actuation"]
    lengths = get_basic_parameters()["lengths"]

    # if plot_peak_alignment:
    x_vals = list()
    y_vals = list()
    for col in df.columns:
        x_vals.append(float(col))
        shoebox_fresh = copy.deepcopy(shoebox)
        actuation_sequence = df[col].array
        postproc_dict = get_postproc_info(shoebox=shoebox_fresh, actuation_sequence=actuation_sequence,
                                          temperature_outside=temperature_outside_series, time_delta=time_delta,
                                          power_weight_curve=power_weight_curve,
                                          temperature_min=temperature_min, temperature_max=temperature_max,
                                          substeps_per_actuation=substeps_per_actuation)
        y_vals.append(postproc_dict["peak_alignment_factor"])
    plt.plot(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel("grid stress index")
    plt.grid(True)
    plt.savefig(filename[:-4] + ".pdf", format="pdf")
    plt.show(block=True)


def pp_from_file(filename, column_index, shoebox, comfort_penalty_weight):
    df = pd.read_csv(filename, index_col=0)
    temperature_outside_series = get_basic_parameters()["temperature_outside_series"]
    time_delta = get_basic_parameters()["time_delta"]
    power_weight_curve = get_basic_parameters()["power_weight_curve"]
    temperature_min = get_basic_parameters()["temperature_min"]
    temperature_max = get_basic_parameters()["temperature_max"]
    substeps_per_actuation = get_basic_parameters()["substeps_per_actuation"]
    # lengths = get_basic_parameters()["lengths"]
    actuation_sequence = df.iloc[:, column_index].array
    postproc_dict = get_postproc_info(shoebox=shoebox, actuation_sequence=actuation_sequence,
                                      temperature_outside=temperature_outside_series, time_delta=time_delta,
                                      power_weight_curve=power_weight_curve,
                                      temperature_min=temperature_min, temperature_max=temperature_max,
                                      substeps_per_actuation=substeps_per_actuation, comfort_penalty_weight=comfort_penalty_weight)
    post_proc(postproc_dict, actuation_sequence=np.repeat(actuation_sequence, substeps_per_actuation), power_weight_curve=power_weight_curve)


if __name__ == "__main__":
    # alternative_main()
    # pp_script("20250509_actuation_results_u0.3_tc3.6e+06_cpwV.csv")
    # pp_script(filename="20250509_actuation_results_uV_tc3.6e+06_cpw1.0e+07.csv", plot_column_index=0, plot_peak_alignment=False,
    #           x_label="u-value in W/m2k")

    # plot_grid_stress_index(filename="20250509_actuation_results_u0.3_tcV_cpw1.0e+07.csv",
    #                        x_label="thermal capacity in J/K")
    #
    # plot_grid_stress_index(filename="20250509_actuation_results_uV_tc3.6e+06_cpw1.0e+07.csv",
    #                        x_label="thermal transmittance in W/m2K")

    # shoebox = ShoeBox(lengths=(5, 5, 5), u_value=.3, therm_sto=3.6e6)
    # pp_from_file(filename="20250508_actuation_results_u0.3_tc3.6e+06_cpwV.csv", column_index=2, shoebox=shoebox, comfort_penalty_weight=1000.)

    # weight_curve = get_consumption_weight_curve(resample_in_minutes=1)
    # plt.plot(weight_curve)
    # plt.grid()
    # ax = plt.gca()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # plt.show(block=True)

    main_script()
