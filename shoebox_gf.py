import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import time
import copy
import pandas as pd
import pickle
from functools import partial

from multiprocessing import Pool

from numba import float64
from numba.experimental import jitclass
from numba import njit
import sys

import pso
import postproc as pp

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
                storage_to_outside = thermal_transmittance_local * (
                        temperature_air_local - temperature_outside_series[global_index])
                Rsi = 0.13  # in m2K/W
                air_to_storage = (temperature_air_local - temperature_storage_local) * (
                        area_hull_local + area_ceiling_local) / Rsi
                dT_air = (heating_to_air - air_to_storage) / capacity_air_local * time_delta
                dT_storage = (
                                     heating_to_storage + air_to_storage - storage_to_outside) / capacity_storage_local * time_delta
                temperature_air_local += dT_air
                temperature_storage_local += dT_storage

                surface_temperature = ((
                                               area_hull_local + area_ceiling_local) * temperature_storage_local + area_floor_local * temperature_supply_local) / (
                                              area_hull_local + area_floor_local + area_ceiling_local)
                operative_temperature = 0.5 * (temperature_air_local + surface_temperature)

                # register results
                result[global_index, 0] = operative_temperature
                result[global_index, 1] = surface_temperature
                result[global_index, 2] = temperature_air_local
                result[global_index, 3] = temperature_storage_local

        self.temperature_air = temperature_air_local
        self.temperature_storage = temperature_storage_local

        return result

    def copy(self):
        '''
        needed this, because the jit compiled shoebox is not pickelable => cannot use deepcopy
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
                                  temperature_operative_series, temperature_max, temperature_min, heating_min=0,
                                  heating_max=6000,
                                  power_penalty_weight=1., comfort_penalty_weight=1.e5, control_penalty_weight=1.e6):
    control_penalty_array = (np.maximum(heating_strategy - heating_max, 0)
                             + np.maximum(heating_min - heating_strategy, 0))

    power_penalty_array = heating_strategy * power_weight_curve
    comfort_penalty_array = (np.maximum(temperature_operative_series - temperature_max, 0)
                             + np.maximum(temperature_min - temperature_operative_series, 0))

    return {"cost_total": (power_penalty_weight * power_penalty_array.sum()
                           + comfort_penalty_weight * comfort_penalty_array.sum()
                           + control_penalty_weight * control_penalty_array.sum()),
            "cost_power": power_penalty_weight * power_penalty_array,
            "cost_comfort": comfort_penalty_weight * comfort_penalty_array,
            "cost_control": control_penalty_weight * control_penalty_array}


def cost_function_pso_slick(heating_strategy, power_weight_curve,
                            temperature_operative_series, temperature_max, temperature_min,
                            power_penalty_weight=1., comfort_penalty_weight=1.e5):
    power_penalty_array = heating_strategy * power_weight_curve
    comfort_penalty_array = (np.maximum(temperature_operative_series - temperature_max, 0)
                             + np.maximum(temperature_min - temperature_operative_series, 0))
    return comfort_penalty_weight * comfort_penalty_array.sum() + power_penalty_weight * power_penalty_array.sum()


def cost_wrapper(heating_strategy, shoebox, temperature_outside_series, time_delta,
                 power_weight_curve, temperature_min, temperature_max, substeps_per_actuation,
                 comfort_penalty_weight=1.e5, control_penalty_weight=1.e6, return_full_dict=False):
    # shoebox_copy = copy.deepcopy(shoebox)
    shoebox_copy = shoebox.copy()
    result_dict = model(shoebox_copy, heating_strategy, temperature_outside_series, time_delta, substeps_per_actuation)
    temperature_operative_series = result_dict['temperature_operative_series']

    cost_dict = cost_function_demand_response(np.repeat(heating_strategy, substeps_per_actuation), power_weight_curve,
                                              temperature_operative_series, temperature_max, temperature_min,
                                              comfort_penalty_weight=comfort_penalty_weight,
                                              control_penalty_weight=control_penalty_weight)
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


def get_basic_parameters():
    temperature_outside = 0
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

    lengths = np.array((10., 6., 3.))
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


def multiproc_wrapper(parameter_dict):
    shoebox = ShoeBox(**parameter_dict['shoebox_parameters'])
    wrapped_func = partial(cost_wrapper, shoebox=shoebox,
                           temperature_outside_series=parameter_dict['simulation_parameters'][
                               'temperature_outside_series'],
                           time_delta=parameter_dict['simulation_parameters']['time_delta'],
                           power_weight_curve=parameter_dict['simulation_parameters']['power_weight_curve'],
                           temperature_min=parameter_dict['simulation_parameters']['temperature_min'],
                           temperature_max=parameter_dict['simulation_parameters']['temperature_max'],
                           substeps_per_actuation=parameter_dict['simulation_parameters']['substeps_per_actuation'],
                           comfort_penalty_weight=parameter_dict['simulation_parameters']['comfort_penalty_weight'],
                           control_penalty_weight=parameter_dict['simulation_parameters']['control_penalty_weight'])

    solution = pso.pso(wrapped_func, dim=24, n_particles=30, n_iters=1000, print_every=None, bounds=(0, 3000),
                       stepsize=500, c1=1, c2=1, randomness=.3, visualize=False, num_processes=1)
    actuation_sequence = solution[0]

    results_dict = {'actuation_sequence': actuation_sequence,
                    'shoebox_parameters': parameter_dict['shoebox_parameters'],
                    'simulation_parameters': parameter_dict['simulation_parameters']}

    return results_dict


def simulation_script(outfile_name, num_vals=20, num_processes=8):
    # Model parameters
    basic_parameter_dict = get_basic_parameters()
    temperature_outside_series = basic_parameter_dict["temperature_outside_series"]
    time_delta = basic_parameter_dict["time_delta"]
    power_weight_curve = basic_parameter_dict["power_weight_curve"]
    temperature_min = basic_parameter_dict["temperature_min"]
    temperature_max = basic_parameter_dict["temperature_max"]
    substeps_per_actuation = basic_parameter_dict["substeps_per_actuation"]

    storage_thickness_array = np.linspace(0.01, 0.05, num_vals)
    insulation_thickness_array = np.linspace(0.05, 0.3, num_vals)

    comfort_penalty_weight = 1.e7
    control_penalty_weight = 1.e6

    # shoebox_init = copy.deepcopy(shoebox)
    results_list = list()

    # assemble list of parameter-tuples
    parameter_list = list()
    for i, storage_thickness in enumerate(storage_thickness_array):
        for j, insulation_thickness in enumerate(insulation_thickness_array):
            insulation_thickness = insulation_thickness_array[j]
            shoebox_parameters = {'length1': 5., 'length2': 5., 'length3': 5., 'heat_max': 6000.,
                                  'storage_thickness': storage_thickness,
                                  'temp_init': 20., 'convective_portion': 0.3,
                                  'insulation_thickness': insulation_thickness}
            simulation_parameters = {'temperature_outside_series': temperature_outside_series,
                                     'time_delta': time_delta, 'power_weight_curve': power_weight_curve,
                                     'temperature_min': temperature_min, 'temperature_max': temperature_max,
                                     'substeps_per_actuation': substeps_per_actuation,
                                     'comfort_penalty_weight': comfort_penalty_weight,
                                     'control_penalty_weight': control_penalty_weight}
            parameter_list.append(
                {'shoebox_parameters': shoebox_parameters, 'simulation_parameters': simulation_parameters})

    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(multiproc_wrapper, parameter_list)):
            end_time = time.time()
            print(
                f'Completed {i + 1} / {len(parameter_list)}, time: {end_time - start_time:.2f} / {(end_time - start_time) * len(parameter_list) / (i + 1):.2f}',
                # end='\r', flush=True)
                flush = True)
            # sys.stdout.flush() chattie suggested this, but it does not work
            results_list.append(result)
    end_time = time.time()
    print(f'\nTime taken: {end_time - start_time:.2f}')

    if outfile_name is not None:
        print(f'saving results to: {outfile_name}')
        with open(outfile_name, 'wb') as f:
            pickle.dump(results_list, f)
    else:
        print('computation done. no results file specified.')


def plot_script(results_file):
    with open(results_file, 'rb') as f:
        data = pickle.load(f)

    if True:
        # add postproc info
        for results_dict in data:
            shoebox_fresh = ShoeBox(**results_dict['shoebox_parameters'])
            postproc_dict = pp.get_postproc_info(shoebox=shoebox_fresh,
                                                 actuation_sequence=results_dict['actuation_sequence'],
                                                 temperature_outside=results_dict['simulation_parameters'][
                                                     'temperature_outside_series'],
                                                 time_delta=results_dict['simulation_parameters']['time_delta'],
                                                 power_weight_curve=results_dict['simulation_parameters'][
                                                     'power_weight_curve'],
                                                 temperature_min=results_dict['simulation_parameters']['temperature_min'],
                                                 temperature_max=results_dict['simulation_parameters']['temperature_max'],
                                                 substeps_per_actuation=results_dict['simulation_parameters'][
                                                     'substeps_per_actuation'],
                                                 comfort_penalty_weight=results_dict['simulation_parameters'][
                                                     'comfort_penalty_weight'])
            results_dict['postproc_dict'] = postproc_dict



    ins_list = list(set([element['shoebox_parameters']['insulation_thickness'] for element in data]))
    ins_list.sort()
    sto_list = list(set([element['shoebox_parameters']['storage_thickness'] for element in data]))
    sto_list.sort()
    # sto_idx = 14
    # d_sto = sto_list[sto_idx]
    # d_ins = [ins_list[5], ins_list[17], ins_list[-1]]

    pp.set_style(font_size=18, font_family='Times New Roman', usetex=True)
    # pp.compare_2runs(data, {'insulation_thickness': (d_ins[1], d_ins[2]), 'storage_thickness': (d_sto, d_sto)}, y_lim=(0, 2000))

    # pp.pp_from_file(data, y_idx=14)
    pp.pp_from_file(data)


def main_script():
    # fn_global = '20250806_results20.pkl'
    # fn_global = '20251021_results30.pkl'
    num_vals = 10
    fn_global = 'dummy_results10.pkl'
    simulation_script(outfile_name=fn_global, num_vals=num_vals, num_processes=15)

    # results_file = '20250915_results30.pkl'
    # results_file = '20251021_results30.pkl'
    results_file = 'dummy_results10.pkl'
    plot_script(results_file=results_file)


if __name__ == "__main__":
    main_script()
