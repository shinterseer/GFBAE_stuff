import numpy as np
import matplotlib.pyplot as plt


class Material:
    def __init__(self, thermal_conductivity: float, density: float, specific_heat: float):
        self.thermal_conductivity = thermal_conductivity
        self.density = density
        self.specific_heat = specific_heat


class Layer:
    def __init__(self, material: Material, thickness: float):
        self.material = material
        self.thickness = thickness


def solve_layers(layers, temperature_left, temperature_right, surface_resistance_left=.13, surface_resistance_right=.04):
    def get_conductivity_between_layers(layer1, layer2):
        """ returns conductivity coefficient between two layers"""
        d1 = layer1.thickness
        d2 = layer2.thickness
        lam1 = layer1.material.thermal_conductivity
        lam2 = layer2.material.thermal_conductivity
        R1 = d1 / 2 / lam1
        R2 = d2 / 2 / lam2
        return 1 / (R1 + R2)

    # setup system (tridiagonal matrix)
    rhs_vector = np.zeros(len(layers))
    K_A1 = 1 / (layers[0].thickness / 2 / layers[0].material.thermal_conductivity + surface_resistance_left)
    K_nB = 1 / (layers[-1].thickness / 2 / layers[-1].material.thermal_conductivity + surface_resistance_right)

    rhs_vector[0] = -K_A1 * temperature_left
    rhs_vector[-1] = -K_nB * temperature_right

    # compute conductivities
    conductivity_vector = np.zeros(len(layers) + 1)  # conductivity vector[i] = conductivity between (i-1, i)
    conductivity_vector[0] = K_A1
    for i in range(1, conductivity_vector.size - 1):
        conductivity_vector[i] = get_conductivity_between_layers(layers[i - 1], layers[i])
    conductivity_vector[-1] = K_nB

    # assemble matrix
    system_matrix = np.zeros((len(layers), len(layers)))
    # write first and last line
    system_matrix[0, 0] = -(conductivity_vector[0] + conductivity_vector[1])
    system_matrix[0, 1] = conductivity_vector[1]
    system_matrix[-1, -2] = conductivity_vector[-2]
    system_matrix[-1, -1] = -(conductivity_vector[-1] + conductivity_vector[-2])

    # write middle lines
    for i in range(1, len(layers) - 1):
        system_matrix[i, i - 1] = conductivity_vector[i]
        system_matrix[i, i] = -(conductivity_vector[i] + conductivity_vector[i + 1])
        system_matrix[i, i + 1] = conductivity_vector[i + 1]

    # solve
    return np.linalg.solve(system_matrix, rhs_vector)


def solve_nodes(layers, temperature_left, temperature_right,
                surface_resistance_left=.13, surface_resistance_right=.04):
    # setup system (tridiagonal matrix)
    rhs_vector = np.zeros(len(layers) + 1)
    rhs_vector[0] = -1 / surface_resistance_left * temperature_left
    rhs_vector[-1] = -1 / surface_resistance_right * temperature_right

    # compute conductivities
    conductivity_vector = np.zeros(len(layers) + 2)  # conductivity vector[i] = conductivity between (i-1, i)
    conductivity_vector[0] = 1 / surface_resistance_left
    conductivity_vector[1:-1] = [l.material.thermal_conductivity / l.thickness for l in layers]
    conductivity_vector[-1] = 1 / surface_resistance_right

    # assemble matrix
    # system_matrix = np.zeros((len(layers), len(layers)))
    main_diagonal = [-(conductivity_vector[i] + conductivity_vector[i + 1]) for i in range(conductivity_vector.size - 1)]
    first_upper_diagonal = [conductivity_vector[i] for i in range(1, conductivity_vector.size - 1)]
    first_lower_diagonal = [conductivity_vector[i] for i in range(1, conductivity_vector.size - 1)]

    system_matrix = np.diag(main_diagonal) + np.diag(first_upper_diagonal, k=1) + np.diag(first_lower_diagonal, k=-1)

    # solve
    return np.linalg.solve(system_matrix, rhs_vector)


def plot_layers(ax, positions, solution):
    print(solution)

    solution = list(solution)
    solution.append(solution[-1])
    # Plot step-wise temperature profile
    ax.step(positions, solution, where='post', label="Solution Layers")


def get_wall():
    plaster = Material(.78, 1600, specific_heat=1000)
    concrete = Material(2.3, 2300, specific_heat=1000)
    rockwool = Material(.04, 40, specific_heat=1030)

    layers = [Layer(plaster, .01),
              Layer(concrete, .075),
              Layer(concrete, .075),
              Layer(rockwool, .1),
              Layer(rockwool, .1),
              Layer(plaster, .01)]
    return layers


def steady_state_script():
    layers = get_wall()

    temperatures_layers = solve_layers(layers, 20, 0,
                                       .13, .04)

    temperatures_nodes = solve_nodes(layers, 20, 0,
                                     .13, .04)

    # fig, ax = plt.subplots()
    ax = plt.gca()
    positions = [0]
    for layer in layers:
        positions.append(positions[-1] + layer.thickness)

    layer_midpoints = [1 / 2 * (positions[i] + positions[i + 1]) for i in range(len(positions) - 1)]
    ax.vlines(positions, ymin=min(temperatures_nodes) - 2, ymax=max(temperatures_nodes) + 2, colors='black', linestyles='dashed', linewidth=0.8)
    ax.plot(layer_midpoints, temperatures_layers, label="Solution Layers")

    # plot_layers(ax, positions, temperatures_layers)
    ax.plot(positions, temperatures_nodes, label="Solution Nodes")
    ax.set_xlabel("Depth [m]")
    ax.set_ylabel("Temperature [°C]")
    plt.grid(True)
    plt.legend()
    plt.show(block=True)


def get_node_capacities(layers):
    node_capacities = np.zeros(len(layers) + 1)
    node_capacities[0] = layers[0].thickness / 2 * layers[0].material.density * layers[0].material.specific_heat
    node_capacities[-1] = layers[-1].thickness / 2 * layers[-1].material.density * layers[-1].material.specific_heat
    node_capacities[1:-1] = [(layers[i].thickness / 2 * layers[i].material.density * layers[i].material.specific_heat +
                              layers[i + 1].thickness / 2 * layers[i + 1].material.density * layers[i + 1].material.specific_heat)
                             for i in range(len(layers) - 1)]
    return node_capacities


def get_conductivity_vector(layers, surface_resistance_left, surface_resistance_right):
    conductivity_vector = np.zeros(len(layers) + 2)  # conductivity vector[i] = conductivity between (i-1, i)
    conductivity_vector[0] = 1 / surface_resistance_left
    conductivity_vector[1:-1] = [l.material.thermal_conductivity / l.thickness for l in layers]
    conductivity_vector[-1] = 1 / surface_resistance_right
    return conductivity_vector


def timestep_nodes(layers, delta_time, node_capacities, temperatures_old, conductivity_vector,
                   temperature_left, temperature_right, surface_resistance_left=.13, surface_resistance_right=.04):
    # setup system (tridiagonal matrix)
    rhs_vector = np.zeros(len(layers) + 1)
    rhs_vector[0] = -1 / surface_resistance_left * temperature_left - temperatures_old[0] * node_capacities[0] / delta_time
    rhs_vector[-1] = -1 / surface_resistance_right * temperature_right - temperatures_old[-1] * node_capacities[-1] / delta_time
    rhs_vector[1:-1] = [(-1) * temperatures_old[i] * node_capacities[i] / delta_time for i in range(1, len(temperatures_old) - 1)]

    # assemble matrix
    # system_matrix = np.zeros((len(layers), len(layers)))
    main_diagonal = [-(conductivity_vector[i] + conductivity_vector[i + 1] + node_capacities[i] / delta_time)
                     for i in range(conductivity_vector.size - 1)]
    first_upper_diagonal = [conductivity_vector[i] for i in range(1, conductivity_vector.size - 1)]
    first_lower_diagonal = [conductivity_vector[i] for i in range(1, conductivity_vector.size - 1)]

    system_matrix = np.diag(main_diagonal) + np.diag(first_upper_diagonal, k=1) + np.diag(first_lower_diagonal, k=-1)

    # solve
    return np.linalg.solve(system_matrix, rhs_vector)


def dynamic_script():
    layers = get_wall()
    temperature_left = 20
    temperature_right = 0
    delta_time = 3600
    surface_resistance_left = .13
    surface_resistance_right = .04
    conductivity_vector = get_conductivity_vector(layers, surface_resistance_left, surface_resistance_right)
    node_capacities = get_node_capacities(layers)
    temperatures_top_start = np.ones(len(node_capacities)) * 20
    temperatures_bottom_start = np.ones(len(node_capacities)) * 0

    temperatures_top = [temperatures_top_start]
    temperatures_bottom = [temperatures_bottom_start]
    temperatures_top.append(timestep_nodes(layers, delta_time, node_capacities, temperatures_top_start, conductivity_vector,
                                           temperature_left, temperature_right, surface_resistance_left=.13, surface_resistance_right=.04))
    temperatures_bottom.append(timestep_nodes(layers, delta_time, node_capacities, temperatures_bottom_start, conductivity_vector,
                                              temperature_left, temperature_right, surface_resistance_left=.13, surface_resistance_right=.04))
    for i in range(50):
        temperatures_top.append(timestep_nodes(layers, delta_time, node_capacities, temperatures_top[-1], conductivity_vector,
                                               temperature_left, temperature_right, surface_resistance_left=.13, surface_resistance_right=.04))
        temperatures_bottom.append(timestep_nodes(layers, delta_time, node_capacities, temperatures_bottom[-1], conductivity_vector,
                                                  temperature_left, temperature_right, surface_resistance_left=.13, surface_resistance_right=.04))

    temperatures_nodes_ss = solve_nodes(layers, 20, 0,
                                        .13, .04)

    positions = [0]
    for layer in layers:
        positions.append(positions[-1] + layer.thickness)

    for i in range(len(temperatures_top)):
        if i % 10 == 0:
            plt.plot(positions, temperatures_top[i], label=str(i), color='black', linewidth=0.5)
            plt.plot(positions, temperatures_bottom[i], label=str(i), color='black', linewidth=0.5)
        else:
            plt.plot(positions, temperatures_top[i], label=str(i), color='grey', linewidth=0.5)
            plt.plot(positions, temperatures_bottom[i], label=str(i), color='grey', linewidth=0.5)

    plt.vlines(positions, ymin=-4, ymax=24, colors='black', linestyles='dashed', linewidth=0.8)
    plt.plot(positions, temperatures_nodes_ss, label="Solution Steady State", linewidth=2)


    plt.xlabel("Depth [m]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)
    # plt.legend()
    plt.show(block=True)


def main():
    dynamic_script()
    # steady_state_script()


if __name__ == "__main__":
    main()
