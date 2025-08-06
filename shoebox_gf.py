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

from multiprocessing import Pool

from numba import float64
from numba.experimental import jitclass
from numba import njit

import pso

spec = [
    ('temp_init', float64),
    ('length1', float64),
    ('length2', float64),
    ('length3', float64),
    ('temperature_supply', float64),
    ('volume', float64),
    ('area_floor', float64),
    ('area_ceiling', float64),
    ('area_hull', float64),
    ('air_density', float64),
    ('air_spec_heat', float64),
    ('capacity_air', float64),
    ('storage_thickness', float64),
    ('capacity_storage', float64),
    ('heat_max', float64),
    ('temperature_storage', float64),
    ('temperature_air', float64),
    ('convective_portion', float64),
    ('u_value', float64),
    ('insulation_thickness', float64),
    ('thermal_transmittance', float64),
]


@jitclass(spec)
class ShoeBox:
    # def __init__(self, length1=5., length2=5., length3=5., heat_max=6000., therm_sto=3.6e6, temp_init=20.,
    def __init__(self, length1=5., length2=5., length3=5., heat_max=6000., storage_thickness=.03, temp_init=20.,
                 convective_portion=0.3, insulation_thickness=0.1):
        self.temp_init = temp_init
        self.length1 = length1
        self.length2 = length2
        self.length3 = length3
        self.temperature_supply = temp_init
        self.volume = length1 * length2 * length3
        self.area_floor = length1 * length2
        self.area_ceiling = length1 * length2
        self.area_hull = 2 * length1 * length3 + 2 * length2 * length3
        self.air_density = 1.2041  # kg/m3
        self.air_spec_heat = 1005.  # J/kgK
        self.capacity_air = self.volume * self.air_spec_heat * self.air_density  # in J/K
        self.storage_thickness = storage_thickness
        # self.capacity_storage = therm_sto  # in J/K

        # reinforced concrete cap = 1000 J/kgK. density = 2500 kg/m3
        self.capacity_storage = self.storage_thickness * (self.area_hull + self.area_ceiling) * 2500. * 1000.  # in J/K
        self.heat_max = heat_max  # in W
        # self.delta_temperature_max = delta_temperature_max
        self.temperature_storage = temp_init
        self.temperature_air = temp_init
        self.convective_portion = convective_portion
        # self.u_value = u_value
        self.insulation_thickness = insulation_thickness
        self.u_value = 0.04 / self.insulation_thickness
        self.thermal_transmittance = self.u_value * (self.area_hull + self.area_floor + self.area_ceiling)

    # def get_surface_temperature(self):
    #     return ((self.area_hull + self.area_ceiling) * self.temperature_storage + self.area_floor * self.temperature_supply) / (self.area_hull + self.area_floor + self.area_ceiling)
    #
    # def get_operative_temperature(self):
    #     return 0.5 * (self.temperature_air + self.get_surface_temperature())
    #     # temp_surf = ((self.area_hull + self.area_ceiling) * self.temperature_storage + self.area_floor * self.temperature_supply) / (self.area_hull + self.area_floor + self.area_ceiling)
    #     # return 0.5 * (self.temperature_air + temp_surf)

    # def timestep(self, heating_power, temperature_outside, time_delta=120):
    #     '''
    #     timestep is not being called anymore - it has all been moved to self.simulation
    #     '''
    #     heating_power = min(self.heat_max, heating_power)
    #     heating_power = max(0, heating_power)
    #
    #     # change temperatures
    #     heating_to_storage = heating_power * (1 - self.convective_portion)
    #     heating_to_air = heating_power * self.convective_portion
    #     storage_to_outside = self.thermal_transmittance * (self.temperature_air - temperature_outside)
    #     Rsi = 0.13  # in m2K/W
    #     air_to_storage = (self.temperature_air - self.temperature_storage) * (self.area_hull + self.area_ceiling) / Rsi
    #     dT_air = (heating_to_air - air_to_storage) / self.capacity_air * time_delta
    #     dT_storage = (heating_to_storage + air_to_storage - storage_to_outside) / self.capacity_storage * time_delta
    #     self.temperature_air += dT_air
    #     self.temperature_storage += dT_storage

    def simulation(self, heating_strategy, temperature_outside_series, time_delta, substeps_per_actuation):
        '''
        This was only written to move the timestepping loop into the class so it would be compiled with numba jit.
        No additional speedup was observed though.
        There is code duplication with timestep(), because I wanted to avoid references to self inside the loop.
        '''

        # init temperature_operative_series
        num_actuation_steps = heating_strategy.size
        result = np.empty((num_actuation_steps * substeps_per_actuation, 4))

        heat_max_local = self.heat_max
        convective_portion_local = self.convective_portion
        thermal_transmittance_local = self.thermal_transmittance
        temperature_air_local = self.temperature_air
        temperature_storage_local = self.temperature_storage
        area_hull_local = self.area_hull
        area_ceiling_local = self.area_ceiling
        capacity_air_local = self.capacity_air
        capacity_storage_local = self.capacity_storage
        temperature_supply_local = self.temperature_supply
        area_floor_local = self.area_floor

        # loop over timesteps
        for i in range(num_actuation_steps):
            heating_power = heating_strategy[i]
            for j in range(substeps_per_actuation):
                global_index = i * substeps_per_actuation + j

                heating_power = min(heat_max_local, heating_power)
                heating_power = max(0., heating_power)

                # change temperatures
                heating_to_storage = heating_power * (1. - convective_portion_local)
                heating_to_air = heating_power * convective_portion_local
                storage_to_outside = thermal_transmittance_local * (temperature_air_local - temperature_outside_series[global_index])
                Rsi = 0.13  # in m2K/W
                air_to_storage = (temperature_air_local - temperature_storage_local) * (area_hull_local + area_ceiling_local) / Rsi
                dT_air = (heating_to_air - air_to_storage) / capacity_air_local * time_delta
                dT_storage = (heating_to_storage + air_to_storage - storage_to_outside) / capacity_storage_local * time_delta
                temperature_air_local += dT_air
                temperature_storage_local += dT_storage

                surface_temperature = ((area_hull_local + area_ceiling_local) * temperature_storage_local + area_floor_local * temperature_supply_local) / (
                        area_hull_local + area_floor_local + area_ceiling_local)
                operative_temperature = 0.5 * (temperature_air_local + surface_temperature)

                # register operative temperature
                result[global_index, 0] = operative_temperature
                result[global_index, 1] = surface_temperature
                result[global_index, 2] = temperature_air_local
                result[global_index, 3] = temperature_storage_local

        self.temperature_air = temperature_air_local
        self.temperature_storage = temperature_storage_local

        return result

    def copy(self):
        '''
        needed this, because the jit compiled shoebox is not pickelable
        '''
        return ShoeBox(self.length1,
                       self.length2,
                       self.length3,
                       self.heat_max,
                       self.storage_thickness,
                       self.temp_init,
                       self.convective_portion,
                       self.insulation_thickness)


def model(shoebox, heating_strategy, temperature_outside_series, time_delta, substeps_per_actuation):
    # result = model_kernel(shoebox, heating_strategy, temperature_outside_series, time_delta, substeps_per_actuation)
    result = shoebox.simulation(heating_strategy, temperature_outside_series, time_delta, substeps_per_actuation)
    # result = simulation(shoebox, heating_strategy, temperature_outside_series, time_delta, substeps_per_actuation)

    # result_dict = {'temperature_operative_series': temperature_operative_series,
    #                'temperature_surface_series': temperature_surface_series,
    #                'temperature_air_series': temperature_air_series,
    #                'temperature_storage_series': temperature_storage_series}
    result_dict = {'temperature_operative_series': result[:, 0],
                   'temperature_surface_series': result[:, 1],
                   'temperature_air_series': result[:, 2],
                   'temperature_storage_series': result[:, 3]}

    return result_dict


# def cost_function_temperature_setpoint(T_history, T_setpoint):
#     """this should now call model to get the history of the operative temperature"""
#     # return sum([(T - T_setpoint) ** 2 for T in T_history[1:]])
#     return np.sum((T_history[1:] - T_setpoint) ** 2)


# def cost_function_demand_response3(heating_strategy, power_weight_curve,
#                                    temperature_operative_series, temperature_max, temperature_min,
#                                    power_penalty_weight=1., comfort_penalty_weight=1.e5):
#     power_penalty = np.dot(heating_strategy, power_weight_curve)
#
#     temperature_mid = 0.5 * (temperature_max + temperature_min)
#     comfort_penalty_array = (temperature_operative_series - temperature_mid) / (temperature_max - temperature_mid)
#     comfort_penalty_array2 = comfort_penalty_array * comfort_penalty_array
#     comfort_penalty_array4 = comfort_penalty_array2 * comfort_penalty_array2
#     comfort_penalty_array8 = comfort_penalty_array4 * comfort_penalty_array4
#     comfort_penalty_array16 = comfort_penalty_array8 * comfort_penalty_array8
#     comfort_penalty_array32 = comfort_penalty_array16 * comfort_penalty_array16
#     comfort_penalty_array64 = comfort_penalty_array32 * comfort_penalty_array32
#     comfort_penalty = comfort_penalty_array64.sum()
#
#     return power_penalty_weight * power_penalty + comfort_penalty_weight * comfort_penalty


def cost_function_demand_response(heating_strategy, power_weight_curve,
                                  temperature_operative_series, temperature_max, temperature_min, heating_min=0, heating_max=6000,
                                  power_penalty_weight=1., comfort_penalty_weight=1.e5, control_penalty_weight=1.e6):
    control_penalty_array = (np.maximum(heating_strategy - heating_max, 0)
                             + np.maximum(heating_min - heating_strategy, 0))
    #
    # heating_strategy = np.maximum(heating_strategy, heating_min)
    # heating_strategy = np.minimum(heating_strategy, heating_max)

    power_penalty_array = heating_strategy * power_weight_curve
    # comfort_penalty_base_array = (temperature_operative_series - 0.5 * (temperature_max + temperature_min)) / comfort_penalty_weight
    # comfort_penalty_soft_array = (np.maximum(temperature_operative_series - temperature_max - .1, 0)
    #                               + np.maximum(temperature_min + .1 - temperature_operative_series, 0)) / np.sqrt(comfort_penalty_weight)
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
    # shoebox_copy = copy.deepcopy(shoebox)
    shoebox_copy = shoebox.copy()
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
    total_energy_turnover = time_delta * np.sum(np.abs(np.repeat(actuation_sequence, substeps_per_actuation)))
    grid_burden = time_delta * np.dot(np.repeat(actuation_sequence, substeps_per_actuation), power_weight_curve)
    grid_stress_index = grid_burden / total_energy_turnover

    result_dict = model(shoebox, heating_strategy=actuation_sequence, temperature_outside_series=temperature_outside,
                        time_delta=time_delta, substeps_per_actuation=substeps_per_actuation)

    cost_dict = cost_function_demand_response(np.repeat(actuation_sequence, substeps_per_actuation), power_weight_curve,
                                              result_dict['temperature_operative_series'], temperature_max, temperature_min,
                                              comfort_penalty_weight=comfort_penalty_weight)

    return {"temperature_air": result_dict['temperature_air_series'],
            "temperature_storage": result_dict['temperature_storage_series'],
            "temperature_operative": result_dict['temperature_operative_series'],
            "temperature_surface": result_dict['temperature_surface_series'],
            "cost_power": cost_dict['cost_power'],
            "cost_comfort": cost_dict['cost_comfort'],
            "cost_control": cost_dict['cost_control'],
            "total_energy_turnover": total_energy_turnover,
            "grid_burden": grid_burden,
            "grid_stress_index": grid_stress_index}


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


def single_simulation_run(wrapped_func, n_particles=30, n_iters=1000, stepsize=500, randomness=.3):
    solution = pso.pso(wrapped_func, dim=24, n_particles=n_particles, n_iters=n_iters, print_every=None, bounds=(0, 3000),
                       stepsize=stepsize, c1=1, c2=1, randomness=randomness, visualize=False, num_processes=1)
    actuation_sequence = solution[0]
    return actuation_sequence


def main_script(outfile_name, plotting=True):
    # Model parameters
    basic_parameter_dict = get_basic_parameters()
    temperature_outside_series = basic_parameter_dict["temperature_outside_series"]
    time_delta = basic_parameter_dict["time_delta"]
    power_weight_curve = basic_parameter_dict["power_weight_curve"]
    # heating_power_initial = basic_parameter_dict["heating_power_initial"]
    # bounds = basic_parameter_dict["bounds"]
    temperature_min = basic_parameter_dict["temperature_min"]
    temperature_max = basic_parameter_dict["temperature_max"]
    substeps_per_actuation = basic_parameter_dict["substeps_per_actuation"]
    lengths = basic_parameter_dict["lengths"]

    # Optimize control actions
    # print("insulation_thickness, storage_thickness, comfort penalty weight, peak alignment factor, total energy turnover, grid burden, cost power, cost comfort, cost_control, computation time in s")
    storage_thickness_list = 0.01 * np.array(list(range(1, 11)))  # 0.01, 0.02, ..., 0.1
    insulation_thickness_list = 0.03 * np.array(list(range(1, 11)))  # 0.03, 0.06, ..., 0.3

    comfort_penalty_weight = 1.e7
    control_penalty_weight = 1.e6

    # shoebox_init = copy.deepcopy(shoebox)
    results_list = list()
    plotting = False

    num_runs = len(storage_thickness_list) * len(insulation_thickness_list)
    for i in range(len(storage_thickness_list)):
        storage_thickness = storage_thickness_list[i]
        for j in range(len(insulation_thickness_list)):
            print(f'run number: {i * len(storage_thickness_list) + j + 1} of {num_runs}')
            insulation_thickness = insulation_thickness_list[j]
            start_time = time.time()
            shoebox_parameters = {'length1': 5., 'length2': 5., 'length3': 5., 'heat_max': 6000., 'storage_thickness': storage_thickness,
                                  'temp_init': 20., 'convective_portion': 0.3, 'insulation_thickness': insulation_thickness}
            shoebox = ShoeBox(**shoebox_parameters)
            wrapped_func = partial(cost_wrapper, shoebox=shoebox, temperature_outside_series=temperature_outside_series,
                                   delta_time=time_delta, power_weight_curve=power_weight_curve,
                                   temperature_min=temperature_min, temperature_max=temperature_max,
                                   substeps_per_actuation=substeps_per_actuation, comfort_penalty_weight=comfort_penalty_weight,
                                   control_penalty_weight=control_penalty_weight, return_full_dict=False)

            actuation_sequence = single_simulation_run(wrapped_func)

            shoebox_fresh = ShoeBox(**shoebox_parameters)
            postproc_dict = get_postproc_info(shoebox=shoebox_fresh, actuation_sequence=actuation_sequence,
                                              temperature_outside=temperature_outside_series, time_delta=time_delta,
                                              power_weight_curve=power_weight_curve,
                                              temperature_min=temperature_min, temperature_max=temperature_max,
                                              substeps_per_actuation=substeps_per_actuation,
                                              comfort_penalty_weight=comfort_penalty_weight)
            computation_time = time.time() - start_time

            results_list.append({'postproc_dict': postproc_dict,
                                 'actuation_sequence': actuation_sequence,
                                 'shoebox_parameters': shoebox_parameters})

    if outfile_name is not None:
        with open(outfile_name, 'wb') as f:
            pickle.dump(results_list, f)
    if plotting:
        post_proc(postproc_dict, actuation_sequence=np.repeat(actuation_sequence, substeps_per_actuation), power_weight_curve=power_weight_curve)


def plot_grid_stress_index(shoebox, filename="20250508_actuation_results_u0.3_tc3.6e+06_cpwV.csv", x_label=None):
    df = pd.read_csv(filename, index_col=0)
    parameter_dict = get_basic_parameters()
    temperature_outside_series = parameter_dict["temperature_outside_series"]
    time_delta = parameter_dict["time_delta"]
    power_weight_curve = parameter_dict["power_weight_curve"]
    temperature_min = parameter_dict["temperature_min"]
    temperature_max = parameter_dict["temperature_max"]
    substeps_per_actuation = parameter_dict["substeps_per_actuation"]
    lengths = parameter_dict["lengths"]

    # if plot_peak_alignment:
    x_vals = list()
    y_vals = list()
    for col in df.columns:
        x_vals.append(float(col))
        shoebox_fresh = shoebox.copy()
        actuation_sequence = df[col].array
        postproc_dict = get_postproc_info(shoebox=shoebox_fresh, actuation_sequence=actuation_sequence,
                                          temperature_outside=temperature_outside_series, time_delta=time_delta,
                                          power_weight_curve=power_weight_curve,
                                          temperature_min=temperature_min, temperature_max=temperature_max,
                                          substeps_per_actuation=substeps_per_actuation)
        y_vals.append(postproc_dict["grid_stress_index"])
    plt.plot(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel("grid stress index")
    plt.grid(True)
    plt.savefig(filename[:-4] + ".pdf", format="pdf")
    plt.show(block=True)


def debugging_stuff(df, data, insulations, storages):
    pwc = get_basic_parameters()['power_weight_curve']
    x=0
    # find the runs
    run1 = dict()
    run2 = dict()
    for run in data:
        if (run['shoebox_parameters']['insulation_thickness'] == insulations[0] and
                run['shoebox_parameters']['storage_thickness'] == storages[0]):
            run1 = run
        if (run['shoebox_parameters']['insulation_thickness'] == insulations[1] and
                run['shoebox_parameters']['storage_thickness'] == storages[1]):
            run2 = run
    x=0


def pp_from_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame([s['shoebox_parameters'] for s in data])
    df['grid_load_index'] = [s['postproc_dict']['grid_stress_index'] for s in data]

    # Pivot to 2D grid format
    pivot = df.pivot(index='storage_thickness', columns='insulation_thickness', values='grid_load_index')

    debugging_stuff(df, data, insulations=[.18, .18], storages=[.04, .08])

    # Create meshgrid from index and columns
    x_vals = pivot.columns.values
    y_vals = pivot.index.values
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Extract z values
    z_vals = pivot.values

    # Find middle indices
    mid_x_idx = len(x_vals) // 2
    mid_y_idx = len(y_vals) // 2

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))

    # --- Plot 1: 3D Surface ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(x_grid, y_grid, z_vals, cmap='viridis')
    ax1.set_title("3D Surface")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    # --- Plot 2: Cross-section at middle x (fixed x, vary y) ---
    ax2 = fig.add_subplot(132)
    ax2.plot(y_vals, z_vals[:, mid_x_idx])
    ax2.set_title(f"Cross-section at x = {x_vals[mid_x_idx]:.2f}")
    ax2.set_xlabel("y")
    ax2.set_ylabel("z")
    ax2.grid(True)

    # --- Plot 3: Cross-section at middle y (fixed y, vary x) ---
    ax3 = fig.add_subplot(133)
    ax3.plot(x_vals, z_vals[mid_y_idx, :])
    ax3.set_title(f"Cross-section at y = {y_vals[mid_y_idx]:.2f}")
    ax3.set_xlabel("x")
    ax3.set_ylabel("z")
    ax3.grid(True)

    plt.tight_layout()
    plt.show(block=True)


def quickplot(myarray):
    plt.plot(myarray)
    plt.grid()
    plt.show(block=True)


if __name__ == "__main__":
    fn_global = '20250806_results.pkl'
    # main_script(outfile_name=None, plotting=True)
    pp_from_file(fn_global)
