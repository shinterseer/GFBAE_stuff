import numpy as np
import matplotlib.pyplot as plt


class Material:
    def __init__(self, thermal_conductivity: float, density: float, specific_heat:float):
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


def plot_layers(ax, positions, layers, solution):
    print(solution)

    solution = list(solution)
    solution.append(solution[-1])
    # Plot step-wise temperature profile
    ax.step(positions, solution, where='post', label="Solution Layers")


def main():
    plaster = Material(.78, 1600, specific_heat=1000)
    concrete = Material(2.3, 2300, specific_heat=1000)
    rockwool = Material(.04, 40, specific_heat=1030)

    layers = [Layer(plaster, .02),
              Layer(concrete, .15),
              Layer(rockwool, .2),
              Layer(plaster, .02)]

    temperatures_layers = solve_layers(layers, 20, 0,
                                       .13, .04)

    temperatures_nodes = solve_nodes(layers, 20, 0,
                                     .13, .04)

    # fig, ax = plt.subplots()

    ax = plt.gca()
    positions = [0]
    for layer in layers:
        positions.append(positions[-1] + layer.thickness)
    ax.vlines(positions, ymin=min(temperatures_nodes) - 2, ymax=max(temperatures_nodes) + 2, colors='gray', linestyles='dashed', linewidth=0.8)

    plot_layers(ax, positions, layers, temperatures_layers)
    ax.plot(positions, temperatures_nodes, label="Solution Nodes")
    ax.set_xlabel("Depth [m]")
    ax.set_ylabel("Temperature [°C]")
    plt.grid(True)
    plt.legend()
    plt.show(block=True)


if __name__ == "__main__":
    main()
