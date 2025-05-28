import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy

def set_global_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# Environment
class SmartBuildingEnv:
    def __init__(self, forecast_len=8, init_temperature=22, temperature_min=20, temperature_max=24, prices=None):
        self.forecast_len = forecast_len
        self.max_steps = 24  # One episode = one day = 24 hours
        self.heating_levels = [0.0, 1.0, 2.0]
        self.init_temperature = init_temperature
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

        # Fixed 24-hour price profile known in advance
        if prices is None:
            self.daily_prices = np.random.uniform(0.1, 0.5, size=24)
        else:
            self.daily_prices = prices
        self.reset()

    def reset(self):
        self.step_count = 0
        self.indoor_temp = self.init_temperature
        self.outdoor_temp = 5
        self.prices = self.daily_prices  # Use the same fixed 24-hour profile
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        # Create a cyclic forecast of 24 values (or self.forecast_len)
        start = self.step_count
        end = start + self.forecast_len

        # Wrap around using modulo
        full_prices = np.concatenate((self.prices, self.prices))  # Ensure safe indexing
        forecast = full_prices[start:end]

        # return np.array([self.indoor_temp, *forecast])
        # return np.array([self._normalize_temperature(self.indoor_temp)])
        return np.array([self.indoor_temp])

    def _normalize_temperature(self, temperature):
        return (temperature - (self.temperature_min + self.temperature_max) / 2) / ((self.temperature_max - self.temperature_min)/2)

    def step(self, action):
        heat_power = self.heating_levels[action]
        heat_loss = (self.indoor_temp - self.outdoor_temp) * 0.075
        self.indoor_temp += heat_power - heat_loss

        price = self.prices[self.step_count]
        energy_cost = price * heat_power

        comfort_penalty = 0.0
        temperature_mid = (self.temperature_min + self.temperature_max) / 2
        comfort_penalty += np.abs(self.indoor_temp - temperature_mid) * 1.
        # if self.indoor_temp < self.temperature_min:
        #     comfort_penalty += (self.temperature_min - self.indoor_temp) * 10
        # if self.indoor_temp > self.temperature_max:
        #     comfort_penalty += (self.indoor_temp - self.temperature_max) * 10

        # reward = -energy_cost - comfort_penalty
        reward = -comfort_penalty

        state = self._get_state()

        self.step_count += 1
        done = False
        if self.step_count >= self.max_steps:
            done = True
            self.reset()

        return state, reward, done


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, num_layers=2, neurons_per_layer=24, learning_rate=0.001,
                 epsilon_decay=0.99, batch_size=32, memory_size=1024):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.model = self._build_model(num_layers, neurons_per_layer)
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self, num_layers, neurons_per_layer):
        model = keras.Sequential()
        input_dim = self.state_size
        output_dim = self.action_size

        # create model
        model.add(layers.Input(shape=(input_dim,)))
        for _ in range(num_layers):
            model.add(layers.Dense(neurons_per_layer, activation='relu'))
        model.add(layers.Dense(output_dim, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        # model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        # for state, action, reward, next_state in minibatch:
        #     target = self.model.predict(state[np.newaxis], verbose=0)
        #     t = self.target_model.predict(next_state[np.newaxis], verbose=0)
        #     target[0][action] = reward + self.gamma * np.amax(t[0])
        #     self.model.fit(state[np.newaxis], target, epochs=1, verbose=0)

        # Vectorized approach
        states = np.array([s for s, _, _, _ in minibatch])
        next_states = np.array([ns for _, _, _, ns in minibatch])
        actions = np.array([a for _, a, _, _ in minibatch])
        rewards = np.array([r for _, _, r, _ in minibatch])

        # Predict Q-values for all states and next states at once
        q_values = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)

        # Compute updated Q-values
        for i, (state, action, reward, next_state) in enumerate(minibatch):
            q_values[i][action] = reward + self.gamma * np.amax(q_next[i])


        # Train in one batch
        self.model.fit(states, q_values, epochs=1, verbose=0)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


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


def training(num_layers, neurons_per_layer, file_model="chatgpt_deep_q_building.keras", num_episodes=300, learning_rate=0.001):
    start_time = time.time()
    prices = get_consumption_weight_curve(resample_in_minutes=60)
    env = SmartBuildingEnv(prices=prices.array)
    state_size = env.reset().shape[0]
    action_size = 3
    epsilon_decay = np.pow(.1, 2 / num_episodes)
    agent = DQNAgent(state_size, action_size, num_layers=num_layers, neurons_per_layer=neurons_per_layer,
                     epsilon_decay=epsilon_decay, learning_rate=learning_rate)
    # agent.epsilon = 0
    # agent.epsilon_min = 0

    for e in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.replay()
        agent.update_target_model()

        print(f"Episode {e + 1}/{num_episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    print(f"Training complete. Time: {time.time() - start_time:.2f}. saving model to {file_model}")
    agent.model.save(file_model)


def test_model(num_layers, neurons_per_layer, file_model="chatgpt_deep_q_building.keras"):
    env = SmartBuildingEnv()
    state_size = env.reset().shape[0]
    action_size = len(env.heating_levels)
    agent = DQNAgent(state_size, action_size, num_layers=num_layers, neurons_per_layer=neurons_per_layer)
    agent.model = keras.models.load_model(file_model)
    agent.epsilon = 0
    agent.epsilon_min = 0
    rewards = np.zeros(24)
    temperatures = np.zeros(24)
    actions = np.zeros(24, dtype=int)

    state = env.reset()
    for i in range(24):
        actions[i] = agent.act(state)
        state, rewards[i], done = env.step(actions[i])
        temperatures[i] = state[0]
        prices = state[1:]

    print(f"rewards over 24 hours: {sum(rewards[:24]):.2f}")
    plt.plot(temperatures, label='Temperature')
    plt.plot(actions, label='Heating level')
    plt.plot(rewards, label='rewards')
    plt.grid()
    plt.legend()
    plt.show(block=True)


def main():
    num_layers = 1
    neurons_per_layer = 3
    set_global_seed()
    training(num_layers, neurons_per_layer, num_episodes=20)
    test_model(num_layers, neurons_per_layer)


if __name__ == '__main__':
    main()
