import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # format datetime axis
import pickle
import pandas as pd
import numpy as np

import shoebox_gf


def array_to_time_series(array, step_in_minutes=1, start_time="2025-04-29 00:00"):
    array = np.array(array)
    step = pd.Timedelta(minutes=step_in_minutes)
    index = pd.date_range(start=pd.Timestamp(start_time), periods=len(array), freq=step)
    return pd.Series(np.array(array), index=index)


def get_postproc_info(shoebox, actuation_sequence, temperature_outside, time_delta, power_weight_curve,
                      temperature_max, temperature_min, substeps_per_actuation, comfort_penalty_weight=1.e5):
    # actuation_sequence = np.repeat(actuation_sequence, substeps_per_actuation)
    total_energy_turnover = time_delta * np.sum(np.abs(np.repeat(actuation_sequence, substeps_per_actuation)))
    grid_burden = time_delta * np.dot(np.repeat(actuation_sequence, substeps_per_actuation), power_weight_curve)
    grid_stress_index = grid_burden / total_energy_turnover

    result_dict = shoebox_gf.model(shoebox, heating_strategy=actuation_sequence, temperature_outside_series=temperature_outside,
                        time_delta=time_delta, substeps_per_actuation=substeps_per_actuation)

    cost_dict = shoebox_gf.cost_function_demand_response(np.repeat(actuation_sequence, substeps_per_actuation), power_weight_curve,
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


def plot_actuation_sequence(actuation_sequence, temperature_operative, axes_object):
    # Temperature plot
    axes_object.set_ylabel('Temperature in °C', color='b')
    axes_object.tick_params(axis='y', labelcolor='b')
    axes_object.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes_object.plot(array_to_time_series(temperature_operative), 'b-', label='Temperature_operative')

    # Control actions plot
    ax_top2 = axes_object.twinx()
    ax_top2.plot(array_to_time_series(actuation_sequence), 'g-', label='Control Actions')
    ax_top2.set_ylabel('actuation sequence (heating power in W)')
    # Adding legends
    axes_object.legend()
    axes_object.grid()


def plot_temperatures(temperature_operative, temperature_air, temperature_surface, temperature_storage, axes_object):
    axes_object.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes_object.plot(array_to_time_series(temperature_operative), label="T_op")
    axes_object.plot(array_to_time_series(temperature_air), label="T_air")
    axes_object.plot(array_to_time_series(temperature_surface), label="T_surf")
    axes_object.plot(array_to_time_series(temperature_storage), label="T_storage")
    axes_object.grid()
    axes_object.legend()


def plot_power_cost(cost_power, power_weight_curve, axes_object):
    axes_object.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes_object.plot(array_to_time_series(cost_power), label="cost power", color="orange")
    axes_object.grid()
    axes_object.legend()
    ax_cost2 = axes_object.twinx()
    ax_cost2.plot(array_to_time_series(power_weight_curve), label="power weight curve")


def plot_comfort(cost_comfort, temperature_operative, axes_object):
    axes_object.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes_object.plot(array_to_time_series(cost_comfort), label="cost comfort", color='orange')
    axes_object.grid()
    axes_object.legend()
    ax_cost_comfort2 = axes_object.twinx()
    ax_cost_comfort2.plot(array_to_time_series(temperature_operative), label="T_op")


def post_proc(postproc_dict, actuation_sequence, power_weight_curve):
    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=2)

    plot_actuation_sequence(actuation_sequence, postproc_dict['temperature_operative'], axes_object=axes[0, 0])
    plot_temperatures(postproc_dict['temperature_operative'], postproc_dict['temperature_air'],
                      postproc_dict['temperature_surface'], postproc_dict['temperature_storage'], axes[1, 0])
    plot_power_cost(postproc_dict["cost_power"], power_weight_curve, axes[0, 1])
    plot_comfort(postproc_dict['cost_comfort'], postproc_dict['temperature_operative'], axes[1, 1])

    plt.tight_layout()
    plt.show(block=True)


def plot_grid_stress_index(shoebox, filename="20250508_actuation_results_u0.3_tc3.6e+06_cpwV.csv", x_label=None):
    df = pd.read_csv(filename, index_col=0)
    parameter_dict = shoebox_gf.get_basic_parameters()
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


def compare_2runs(data, property_dict):
    pwc = shoebox_gf.get_basic_parameters()['power_weight_curve']

    # find the runs
    pkeys = list(property_dict.keys())
    for run in data:
        if (run['shoebox_parameters'][pkeys[0]] == property_dict[pkeys[0]][0] and
                run['shoebox_parameters'][pkeys[1]] == property_dict[pkeys[1]][0]):
            run1 = run
        if (run['shoebox_parameters'][pkeys[0]] == property_dict[pkeys[0]][1] and
                run['shoebox_parameters'][pkeys[1]] == property_dict[pkeys[1]][1]):
            run2 = run

    if run1 is None or run2 is None:
        print('not all runs found')
        return

    # plot
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    plot_actuation_sequence(np.repeat(run1['actuation_sequence'], 60), run1['postproc_dict']['temperature_operative'], axes_object=axs[0, 0])
    plot_power_cost(run1['postproc_dict']["cost_power"], pwc, axs[0, 1])
    plot_actuation_sequence(np.repeat(run2['actuation_sequence'], 60), run2['postproc_dict']['temperature_operative'], axes_object=axs[1, 0])
    plot_power_cost(run2['postproc_dict']["cost_power"], pwc, axs[1, 1])
    plt.tight_layout()
    plt.show(block=True)


def pp_from_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame([s['shoebox_parameters'] for s in data])
    df['grid_load_index'] = [s['postproc_dict']['grid_stress_index'] for s in data]

    # Pivot to 2D grid format
    pivot = df.pivot(index='storage_thickness', columns='insulation_thickness', values='grid_load_index')

    # compare_2runs(data, {'insulation_thickness': (0.18, 0.18), 'storage_thickness': (.04, .08)})

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
