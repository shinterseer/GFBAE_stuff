import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt


# Cost function
def cost_function(u, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out):
    T_predicted = T_initial
    cost = 0
    T_history = [T_initial]
    for k in range(prediction_horizon):
        if k < control_horizon:
            T_predicted = T_predicted + a * u[k] + b * (T_out - T_predicted)
        else:
            T_predicted = T_predicted + a * u[-1] + b * (T_out - T_predicted)  # Assume u stays constant beyond control horizon
        T_history.append(T_predicted)
        cost += (T_predicted - T_setpoint) ** 2
    return cost, T_history

def plotting(optimal_control_actions, T_history, prediction_horizon, T_setpoint):

    # Plotting
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(prediction_horizon + 1)
    plt.plot(time_steps, T_history, 'b-', label='Temperature')
    plt.axhline(y=T_setpoint, color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.title('Predicted Temperature Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()



def plotting2(optimal_control_actions, T_history, prediction_horizon, control_horizon, T_setpoint):
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
    plt.show()

def main_script():
    # Model parameters
    a = 2
    b = 0.1
    T_out = 7  # Outdoor temperature

    # Initial conditions
    T_initial = 18  # Initial room temperature
    T_setpoint = 22  # Desired room temperature

    # for i in range(3):
    # Prediction and control horizons
    prediction_horizon = 60
    control_horizon = 40

    # Constraints
    bounds = [(0, 1)] * control_horizon

    # Initial control actions
    u_initial = [0.5] * control_horizon

    # Optimize control actions

    start_time = time.time()
    result = minimize(lambda u: cost_function(u, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out)[0],
                      u_initial, bounds=bounds)

    # Optimal control actions
    # Optimal control actions
    optimal_control_actions = result.x

    # Calculate the temperature history with optimal control actions
    _, T_history = cost_function(optimal_control_actions, T_initial, T_setpoint, prediction_horizon, control_horizon, a, b, T_out)

    print("Optimal Control Actions:", optimal_control_actions)
    print(f"time taken: {time.time() - start_time:.2f} seconds")

    plotting2(optimal_control_actions, T_history, prediction_horizon, control_horizon, T_setpoint)


if __name__ == "__main__":
    # alternative_main()
   main_script()
