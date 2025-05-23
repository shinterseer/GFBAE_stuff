import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random


# Environment
class SmartBuildingEnv:
    def __init__(self, forecast_len=3, max_steps=48):
        self.forecast_len = forecast_len
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.step_count = 0
        self.indoor_temp = 20.0
        self.outdoor_temp = 5.0
        self.prices = np.random.uniform(0.1, 0.5, size=self.max_steps + self.forecast_len)
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        forecast = self.prices[self.step_count: self.step_count + self.forecast_len]
        return np.array([self.indoor_temp, self.prices[self.step_count], *forecast])

    def step(self, action):
        heat_power = [0.0, 1.0, 2.0][action]
        heat_loss = (self.indoor_temp - self.outdoor_temp) * 0.1
        self.indoor_temp += heat_power - heat_loss

        price = self.prices[self.step_count]
        energy_cost = price * heat_power

        comfort_penalty = 0.0
        if self.indoor_temp < 19:
            comfort_penalty = -1.0
        elif self.indoor_temp > 24:
            comfort_penalty = -0.5

        reward = -energy_cost + comfort_penalty

        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._get_state(), reward, done


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.batch_size = 32
        self.model = self._build_model()
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer='adam')
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


def training(file_model="chatgpt_deep_q_building.keras", num_episodes=300):
    env = SmartBuildingEnv()
    state_size = env.reset().shape[0]
    action_size = 3
    agent = DQNAgent(state_size, action_size)

    for e in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for time in range(env.max_steps):
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

    print(f"Training complete. saving model to {file_model}")
    agent.model.save(file_model)


def test_model(file_model="chatgpt_deep_q_building.keras"):
    env = SmartBuildingEnv()
    state_size = env.reset().shape[0]
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.model = keras.models.load_model(file_model)
    x = 0


# Training loop
if __name__ == '__main__':
    # training(num_episodes=10)
    test_model()
