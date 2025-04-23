import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt


class ShoeBox:
    def __init__(self, lengths, heat_max=2000, delta_temperature_max=40, therm_sto=3.6e6, temp_init=20,
                 convective_portion=0.5, u_value=0.5):
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

    def get_operative_temperature(self, temperature_supply):
        temp_surf = ((self.area_hull + self.area_ceiling) * self.temperature_storage + self.area_floor * temperature_supply) / (self.area_hull + self.area_floor + self.area_ceiling)
        return 0.5 * (self.temperature_air + temp_surf)

    def timestep(self, temperature_outside, temperature_supply, delta_time=60):
        heating_power = (temperature_supply - self.get_operative_temperature(temperature_supply)) / self.delta_temperature_max * self.heat_max
        heating_to_storage = heating_power * self.convective_portion
        heating_to_air = heating_power * (1 - self.convective_portion)
        storage_to_outside = self.thermal_transmittance * (self.temperature_air - temperature_outside)
        Rsi = 0.13  # in m2K/W
        air_to_storage = (self.temperature_air - self.temperature_storage) * (self.area_hull + self.area_ceiling) / Rsi
        dT_air = (heating_to_air - air_to_storage) / self.capacity_air * delta_time
        dT_storage = (heating_to_storage + air_to_storage - storage_to_outside) / self.capacity_storage * delta_time
        self.temperature_air += dT_air
        self.temperature_storage += dT_storage


def model_old(T_predicted, u, a, b, T_out):
    T_new = T_predicted + a * u + b * (T_out - T_predicted)
    return T_new


def model(shoebox, temperature_supply, temperature_outside):
    shoebox.timestep(temperature_supply, temperature_outside)
    return shoebox.get_operative_temperature(temperature_supply)


def cost_function(T_history, T_setpoint):
    return sum([(T - T_setpoint) ** 2 for T in T_history[1:]])


def mpc(shoebox, temperature_supply, T_setpoint, temperature_outside, prediction_horizon, control_horizon):
    T_history = np.empty(prediction_horizon + 1)
    T_initial = shoebox.get_operative_temperature(temperature_supply[0])
    T_history[0] = T_initial
    T_predicted = T_initial

    for k in range(control_horizon):
        # T_predicted = model(T_predicted, u[k], a, b, T_out)
        T_predicted = model(shoebox, temperature_supply[k], temperature_outside)
        T_history[k + 1] = T_predicted
    for k in range(control_horizon, prediction_horizon):
        # T_predicted = model(T_predicted, u[-1], a, b, T_out)  # Assume u stays constant beyond control horizon
        T_predicted = model(shoebox, temperature_supply[-1], temperature_outside)
        T_history[k + 1] = T_predicted

    cost = cost_function(T_history, T_setpoint)
    return cost, T_history


def plotting(optimal_control_actions, T_history, prediction_horizon, control_horizon, T_setpoint):
    # Plotting
    fig, ax1 = plt.subplots()

    # Temperature plot
    time_steps = np.arange(prediction_horizon + 1)
    ax1.plot(time_steps, T_history, 'b-', label='Temperature')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature (°C)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.axhline(y=T_setpoint, color='r', linestyle='--', label='Setpoint')

    # Control actions plot
    ax2 = ax1.twinx()
    control_steps = np.arange(control_horizon)
    ax2.step(control_steps, optimal_control_actions, 'g-', where='mid', label='Control Actions')
    ax2.set_ylabel('Control Actions', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Adding legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Temperature and Control Actions Over Time')
    ax1.grid()
    plt.show()


def main_script():
    # Model parameters
    lengths = (5, 5, 5)
    shoebox = ShoeBox(lengths=lengths)
    temperature_outside = 7

    # Initial conditions
    T_setpoint = 23  # Desired room temperature

    # for i in range(3):
    # Prediction and control horizons
    prediction_horizon = 100
    control_horizon = 50

    # Constraints
    bounds = [(15, 35)] * control_horizon

    # Initial control actions
    # u_initial = [20] * control_horizon
    temperature_supply_initial = np.array([15] * control_horizon)

    # Optimize control actions

    start_time = time.time()

    # def mpc(shoebox, temperature_suppy, T_setpoint, temperature_outside, prediction_horizon, control_horizon, ):
    # result = minimize(lambda u: mpc(u, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out)[0],
    #                       u_initial, bounds=bounds)

    result = minimize(lambda temperature_supply: mpc(shoebox, temperature_supply, T_setpoint, temperature_outside,
                                                     prediction_horizon, control_horizon)[0],
                      temperature_supply_initial, bounds=bounds)

    # Optimal control actions
    # Optimal control actions
    optimal_control_actions = result.x

    # Calculate the temperature history with optimal control actions
    # _, T_history = mpc(optimal_control_actions, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out)
    _, T_history = mpc(shoebox, optimal_control_actions, T_setpoint, temperature_outside,
        prediction_horizon, control_horizon)

    print("Optimal Control Actions:", optimal_control_actions)
    print(f"time taken: {time.time() - start_time:.2f} seconds")

    plotting(optimal_control_actions, T_history, prediction_horizon, control_horizon, T_setpoint)


if __name__ == "__main__":
    # alternative_main()
    main_script()
