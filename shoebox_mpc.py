import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
import copy


class ShoeBox:
    def __init__(self, lengths, heat_max=6000, delta_temperature_max=40, therm_sto=3.6e6, temp_init=20,
                 convective_portion=0.5, u_value=0.5, temperature_supply_delta_max=0.05):
        self.temperature_supply = temp_init
        self.temperature_supply_delta_max = temperature_supply_delta_max
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

    def get_operative_temperature(self):
        temp_surf = ((self.area_hull + self.area_ceiling) * self.temperature_storage + self.area_floor * self.temperature_supply) / (self.area_hull + self.area_floor + self.area_ceiling)
        return 0.5 * (self.temperature_air + temp_surf)

    def timestep(self, temperature_supply_change_signal, temperature_outside, delta_time=120):

        # change supply temperature
        # if abs(self.temperature_supply - temperature_supply_setpoint) < self.temperature_supply_delta_max:
        #     self.temperature_supply = temperature_supply_setpoint
        # elif self.temperature_supply <= temperature_supply_setpoint - self.temperature_supply_delta_max:
        #     self.temperature_supply += self.temperature_supply_delta_max
        # elif self.temperature_supply >= temperature_supply_setpoint + self.temperature_supply_delta_max:
        #     self.temperature_supply -= self.temperature_supply_delta_max

        self.temperature_supply += temperature_supply_change_signal * self.temperature_supply_delta_max

        # change temperatures
        heating_power = (self.temperature_supply - self.get_operative_temperature()) / self.delta_temperature_max * self.heat_max
        heating_to_storage = heating_power * self.convective_portion
        heating_to_air = heating_power * (1 - self.convective_portion)
        storage_to_outside = self.thermal_transmittance * (self.temperature_air - temperature_outside)
        Rsi = 0.13  # in m2K/W
        air_to_storage = (self.temperature_air - self.temperature_storage) * (self.area_hull + self.area_ceiling) / Rsi
        dT_air = (heating_to_air - air_to_storage) / self.capacity_air * delta_time
        dT_storage = (heating_to_storage + air_to_storage - storage_to_outside) / self.capacity_storage * delta_time
        self.temperature_air += dT_air
        self.temperature_storage += dT_storage


def model(shoebox, temperature_supply_change_signal, temperature_outside, time_delta):
    shoebox.timestep(temperature_supply_change_signal, temperature_outside, time_delta)
    return shoebox.get_operative_temperature()


def cost_function(T_history, T_setpoint):
    return np.sum((T_history[1:] - T_setpoint) ** 2)


def mpc_inner(actuation_strategy, shoebox, T_setpoint, temperature_outside, prediction_horizon, control_horizon, time_delta):
    shoebox_copy = copy.deepcopy(shoebox)
    # shoebox_copy = copy.copy(shoebox) # funny enough: shallow copy is slower here
    T_history = np.empty(prediction_horizon + 1)
    # temperature_supply = actuation_strategy[0]
    T_initial = shoebox_copy.get_operative_temperature()
    T_history[0] = T_initial
    T_predicted = T_initial

    for k in range(control_horizon):
        # T_predicted = model(T_predicted, u[k], a, b, T_out)
        T_history[k + 1] = model(shoebox_copy, actuation_strategy[k], temperature_outside, time_delta=time_delta)
    for k in range(control_horizon, prediction_horizon):
        # T_predicted = model(T_predicted, u[-1], a, b, T_out)  # Assume u stays constant beyond control horizon
        T_history[k + 1] = model(shoebox_copy, actuation_strategy[-1], temperature_outside, time_delta=time_delta)

    cost = cost_function(T_history, T_setpoint)
    # return cost, T_history
    return cost


def mpc_outer(num_timesteps, initial_control, bounds, shoebox, T_setpoint,
              temperature_outside, prediction_horizon, control_horizon, delta_time):
    optimal_control_actions = list()
    start_time = time.time()
    for i in range(num_timesteps):
        if i % 50 == 0:
            print("\r", end="")
            print(f"timestep: {i}/{num_timesteps}, total time so far: {time.time() - start_time:.2f}", flush=True, end="")
        # print(f"timestep", flush=True)

        result = minimize(mpc_inner, initial_control, #method="Powell",
                          args=(shoebox, T_setpoint, temperature_outside,
                                prediction_horizon, control_horizon, delta_time),
                          bounds=bounds)

        optimal_control_actions.append(result.x[0])
        shoebox.timestep(optimal_control_actions[-1], temperature_outside)
    print("")

    return np.array(optimal_control_actions)


def get_property_series(shoebox, actuation_strategy, temperature_outside):
    dim = len(actuation_strategy)
    temperature_storage = np.empty(dim + 1)
    temperature_air = np.empty(dim + 1)
    temperature_operative = np.empty(dim + 1)
    temperature_supply = np.empty(dim + 1)

    temperature_storage[0] = shoebox.temperature_storage
    temperature_air[0] = shoebox.temperature_air
    temperature_operative[0] = shoebox.get_operative_temperature()
    temperature_supply[0] = shoebox.temperature_air

    for k in range(dim):
        u = actuation_strategy[k]
        te = temperature_outside[k]
        # shoebox.timestep(te, u)
        shoebox.timestep(u, te)
        temperature_storage[k + 1] = shoebox.temperature_storage
        temperature_air[k + 1] = shoebox.temperature_air
        temperature_operative[k + 1] = shoebox.get_operative_temperature()
        temperature_supply[k + 1] = shoebox.temperature_supply

    return {"temperature_air": temperature_air,
            "temperature_storage": temperature_storage,
            "temperature_operative": temperature_operative,
            "temperature_supply": temperature_supply}

def post_proc(optimal_control_actions, num_timesteps, T_setpoint, lengths, temperature_outside):

    shoebox = ShoeBox(lengths)
    property_dict = get_property_series(shoebox, actuation_strategy=optimal_control_actions,
                                        temperature_outside=temperature_outside)
    # Plotting
    fig, ax1 = plt.subplots()

    # Temperature plot
    time_steps = np.arange(num_timesteps + 1)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature (°C)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.axhline(y=T_setpoint, color='r', linestyle='--', label='Setpoint')
    ax1.step(np.arange(num_timesteps), optimal_control_actions, 'g-', where='mid', label='Control Actions')
    ax1.plot(time_steps, property_dict["temperature_operative"], 'b-', label='Temperature_operative')
    ax1.plot(time_steps, property_dict["temperature_supply"], color='orange', label='Temperature_supply')

    # Control actions plot
    # ax2 = ax1.twinx()
    # control_steps = np.arange(num_timesteps)
    # ax2.step(control_steps, optimal_control_actions, 'g-', where='mid', label='Control Actions')
    # ax2.set_ylabel('Control Actions', color='g')
    # ax2.tick_params(axis='y', labelcolor='g')

    # Adding legends
    ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    plt.title('Temperature and Control Actions Over Time')
    ax1.grid()
    plt.show()


def main_script():
    # Model parameters
    lengths = (5, 5, 5)
    shoebox = ShoeBox(lengths=lengths)
    temperature_outside = 7
    simulated_time = 8 * 3600
    delta_time = 60
    num_timesteps = int(simulated_time / delta_time)

    # Initial conditions
    T_setpoint = 23  # Desired room temperature

    # for i in range(3):
    # Prediction and control horizons
    prediction_horizon = 100
    control_horizon = 5

    # Constraints
    # bounds = [(15, 35)] * control_horizon
    bounds = [(-1, 1)] * control_horizon

    # Initial control actions
    temperature_supply_initial = np.array([20] * control_horizon)

    # Optimize control actions
    start_time = time.time()

    optimal_control_actions = mpc_outer(num_timesteps, temperature_supply_initial, bounds, shoebox, T_setpoint,
                                        temperature_outside, prediction_horizon, control_horizon, delta_time)

    # Calculate the temperature history with optimal control actions
    # _, T_history = mpc(optimal_control_actions, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out)

    shoebox_new = ShoeBox(lengths=lengths)
    temperature_outside_array = np.ones(num_timesteps + 1) * temperature_outside
    # optimal_control_actions = np.ones(optimal_control_actions.size) * 7
    result_dict = get_property_series(shoebox=shoebox_new, actuation_strategy=optimal_control_actions,
                                      temperature_outside=temperature_outside_array)
    T_history = result_dict["temperature_operative"]
    # _, T_history = mpc(optimal_control_actions, shoebox, T_setpoint, temperature_outside,
    #                    prediction_horizon, control_horizon)

    if False:
        print("Optimal Control Actions:", optimal_control_actions)
    print(f"time taken: {time.time() - start_time:.2f} seconds")

    post_proc(optimal_control_actions, num_timesteps, T_setpoint, lengths,
              temperature_outside=np.full(num_timesteps, temperature_outside))


if __name__ == "__main__":
    # alternative_main()
    main_script()
