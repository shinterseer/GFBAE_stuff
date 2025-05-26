import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import pandas as pd
import matplotlib.pyplot as plt
import time

def set_global_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# Environment
class SmartBuildingEnv:
    def __init__(self, forecast_len=8, prices=None):
        self.forecast_len = forecast_len
        self.max_steps = 24  # One episode = one day = 24 hours
        self.heating_levels = [0.0, 1.0, 2.0]

        # Fixed 24-hour price profile known in advance
        if prices is None:
            self.daily_prices = np.random.uniform(0.1, 0.5, size=24)
        else:
            self.daily_prices = prices
        self.reset()

    def reset(self):
        self.step_count = 0
        self.indoor_temp = 20.0
        self.outdoor_temp = 5.0
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

        return np.array([self.indoor_temp, self.prices[self.step_count], *forecast])

    def step(self, action):
        heat_power = self.heating_levels[action]
        heat_loss = (self.indoor_temp - self.outdoor_temp) * 0.05
        self.indoor_temp += heat_power - heat_loss

        price = self.prices[self.step_count]
        energy_cost = price * heat_power

        comfort_penalty = 0.0
        if self.indoor_temp < 19:
            comfort_penalty = -10
        elif self.indoor_temp > 24:
            comfort_penalty = -5

        reward = -energy_cost + comfort_penalty

        self.step_count = (self.step_count + 1) % self.max_steps

        done = self.step_count >= self.max_steps
        return self._get_state(), reward, done


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, num_layers=2, neurons_per_layer=24):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.batch_size = 32
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

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state[np.newaxis], verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state[np.newaxis], verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])
            self.model.fit(state[np.newaxis], target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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


def training(file_model="chatgpt_deep_q_building.keras", num_episodes=300):

    start_time = time.time()
    prices = get_consumption_weight_curve(resample_in_minutes=60)
    env = SmartBuildingEnv(prices=prices.array)
    state_size = env.reset().shape[0]
    action_size = 3
    agent = DQNAgent(state_size, action_size, num_layers=2, neurons_per_layer=24)

    for e in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.replay()
        agent.update_target_model()

        print(f"Episode {e + 1}/{num_episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    print(f"Training complete. Time: {time.time() - start_time:.2f}. saving model to {file_model}")
    agent.model.save(file_model)


def test_model(file_model="chatgpt_deep_q_building.keras"):
    env = SmartBuildingEnv()
    state_size = env.reset().shape[0]
    action_size = len(env.heating_levels)
    agent = DQNAgent(state_size, action_size)
    agent.model = keras.models.load_model(file_model)
    agent.epsilon = 0
    rewards = np.zeros(24 * 4)
    temperatures = np.zeros(24 * 4)
    actions = np.zeros(24 * 4, dtype=int)

    state = env.reset()
    for i in range(24 * 4):
        actions[i] = agent.act(state)
        state, rewards[i], done = env.step(actions[i])
        temperatures[i] = state[0]
        price = state[1]
        forecast = state[2:]

    plt.plot(temperatures, label='Temperature')
    plt.plot(actions, label='Heating level')
    plt.plot(rewards, label='rewards')
    plt.grid()
    plt.legend()
    plt.show(block=True)


# Training loop
if __name__ == '__main__':
    set_global_seed()
    # training(num_episodes=30)
    test_model()
