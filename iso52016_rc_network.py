import numpy as np
import matplotlib.pyplot as plt


class Material:
    def __init__(self, thermal_conductivity, density, specific_heat):
        self.thermal_conductivity = thermal_conductivity
        self.density = density
        self.specific_heat = specific_heat


class Layer:
    def __init__(self, material, thickness):
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
    rhs_vector[0] = -1 / surface_resistance_left * temperature_left
    rhs_vector[-1] = -1 / surface_resistance_right * temperature_right

    # compute conductivities
    conductivity_vector = np.zeros(len(layers) + 1)  # conductivity vector[i] = conductivity between (i-1, i)
    conductivity_vector[0] = 1 / surface_resistance_left
    for i in range(1, conductivity_vector.size - 1):
        conductivity_vector[i] = get_conductivity_between_layers(layers[i - 1], layers[i])
    conductivity_vector[-1] = 1 / surface_resistance_right

    # assemble matrix
    system_matrix = np.zeros(layers.size, layers.size)
    for i in range(layers.size):
        pass

    # solve

    pass


def main():
    pass

    plaster = Material(.78, 1600, specific_heat=1000)
    concrete = Material(2.3, 2300, specific_heat=1000)
    rockwool = Material(.04, 40, specific_heat=1030)

    layers = [Layer(plaster, .02),
              Layer(concrete, .15),
              Layer(rockwool, .2),
              Layer(plaster, .02)]


if __name__ == "__main__":
    main()
