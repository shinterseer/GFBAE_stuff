import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt


def model(T_predicted, u, a, b, T_out, delta_t=1):
    T_new = T_predicted + a * u + b * (T_out - T_predicted) * delta_t
    return T_new


def cost_function(T_history, T_setpoint):
    return sum([(T - T_setpoint) ** 2 for T in T_history[1:]])


def mpc(u, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out):
    T_history = np.empty(prediction_horizon + 1)
    T_history[0] = T_initial
    T_predicted = T_initial

    for k in range(control_horizon):
        T_predicted = model(T_predicted, u[k], a, b, T_out)
        T_history[k + 1] = T_predicted
    for k in range(control_horizon, prediction_horizon):
        T_predicted = model(T_predicted, u[-1], a, b, T_out)  # Assume u stays constant beyond control horizon
        T_history[k + 1] = T_predicted

    cost = cost_function(T_history, T_setpoint)
    return cost, T_history


def mpc_cost(u, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out):
    return mpc(u, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out)[0]


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
    a = 0.4
    b = 0.02
    T_out = 7  # Outdoor temperature

    # Initial conditions
    T_initial = 18  # Initial room temperature
    T_setpoint = 22  # Desired room temperature

    # for i in range(3):
    # Prediction and control horizons
    prediction_horizon = 100
    control_horizon = 50

    # Constraints
    bounds = [(0, 1)] * control_horizon

    # Initial control actions
    u_initial = np.array([0.5] * control_horizon)

    # Optimize control actions
    start_time = time.time()
    # result = minimize(lambda u: mpc(u, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out)[0],
    #                   u_initial, bounds=bounds)
    result = minimize(mpc_cost, u_initial, args=(T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out),
                      bounds=bounds)

    # Optimal control actions
    # Optimal control actions
    optimal_control_actions = result.x

    # Calculate the temperature history with optimal control actions
    _, T_history = mpc(optimal_control_actions, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out)

    print("Optimal Control Actions:", optimal_control_actions)
    print(f"time taken: {time.time() - start_time:.2f} seconds")

    plotting(optimal_control_actions, T_history, prediction_horizon, control_horizon, T_setpoint)


if __name__ == "__main__":
    main_script()
