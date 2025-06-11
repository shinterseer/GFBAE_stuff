# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import backend
# from tensorflow.keras.optimizers import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import deque
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy
import os
import training_runs


def set_global_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)

# Environment
class SmartBuildingEnv:
    def __init__(self, forecast_len=12, init_temperature=22, temperature_min=20, temperature_max=24,
                 heat_power_factor=1, heat_loss_factor=0.025, prices=None,
                 comfort_violation_reward=-100):
        self.forecast_len = forecast_len
        self.max_steps = 24  # One episode = one day = 24 hours
        self.heating_levels = [0.0, 1.0, 2.0]
        self.init_temperature = init_temperature
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.heat_power_factor = heat_power_factor
        self.heat_loss_factor = heat_loss_factor
        self.comfort_violation_reward = comfort_violation_reward

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

        return np.array([self.indoor_temp, *forecast])
        # return np.array([self.indoor_temp])

    def _normalize_temperature(self, temperature):
        return (temperature - (self.temperature_min + self.temperature_max) / 2) / ((self.temperature_max - self.temperature_min) / 2)

    def step(self, action):
        heating_power = self.heating_levels[action] * self.heat_power_factor
        heat_loss = (self.indoor_temp - self.outdoor_temp) * self.heat_loss_factor
        self.indoor_temp += heating_power - heat_loss

        price = self.prices[self.step_count]
        energy_cost = price * heating_power

        # comfort_penalty = 0.0
        # temperature_mid = (self.temperature_min + self.temperature_max) / 2
        # comfort_penalty += np.abs(self.indoor_temp - temperature_mid) * 1.
        # if self.indoor_temp < self.temperature_min:
        #     comfort_penalty += (self.temperature_min - self.indoor_temp) * 10
        # if self.indoor_temp > self.temperature_max:
        #     comfort_penalty += (self.indoor_temp - self.temperature_max) * 10

        # reward = -energy_cost * 30 - comfort_penalty
        # reward = -comfort_penalty
        reward = -energy_cost

        state = self._get_state()

        self.step_count += 1
        done = False
        if self.step_count >= self.max_steps:
            done = True
            self.reset()
        if self.indoor_temp < self.temperature_min or self.indoor_temp > self.temperature_max:
            done = True
            reward = self.comfort_violation_reward
            # self.reset() # do not reset here - break will be hit on the outside if done is true

        return state, reward, done


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, num_layers=2, neurons_per_layer=24, learning_rate=0.001,
                 epsilon_decay=0.99, batch_size=None, memory_size=1024 * 1024):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        if batch_size is None:
            self.batch_size = int(memory_size / 32)
        else:
            self.batch_size = batch_size

        self.model = self._build_model(num_layers, neurons_per_layer)
        # self.target_model = keras.models.clone_model(self.model)
        # self.target_model.set_weights(self.model.get_weights())
        self.target_model = copy.deepcopy(self.model)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)

    # def _build_model(self, num_layers, neurons_per_layer):
    #     model = keras.Sequential()
    #     input_dim = self.state_size
    #     output_dim = self.action_size
    #
    #     # create model
    #     model.add(layers.Input(shape=(input_dim,)))
    #     for _ in range(num_layers):
    #         model.add(layers.Dense(neurons_per_layer, activation='relu'))
    #     model.add(layers.Dense(output_dim, activation='linear'))
    #     model.compile(optimizer='adam', loss='mse')
    #     # model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
    #     return model

    #pytorch version
    def _build_model(self, num_layers, neurons_per_layer):
        input_dim = self.state_size
        output_dim = self.action_size

        layers_pytorch = [nn.Linear(input_dim, neurons_per_layer), nn.ReLU()]

        for _ in range(num_layers - 1):
            layers_pytorch.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers_pytorch.append(nn.ReLU())

        layers_pytorch.append(nn.Linear(neurons_per_layer, output_dim))  # output layer

        model = nn.Sequential(*layers_pytorch)
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # keras/tf version
        # act_values = self.model(state[np.newaxis], verbose=0, training=False)
        # pytorch version
        act_values = self.model(torch.from_numpy(state).unsqueeze(0).float()).detach()
        return np.argmax(act_values[0])

    # def act_batch(self, states: np.ndarray) -> np.ndarray:
    #     # states: shape (batch_size, state_dim)
    #     if np.random.rand() < self.epsilon:
    #         # random actions
    #         return np.random.randint(0, self.action_size, size=len(states))
    #     act_values = self.model(states, training=False)
    #     return np.argmax(act_values.numpy(), axis=1)

    # pytorch version
    def act_batch(self, states: np.ndarray) -> np.ndarray:
        # states: shape (batch_size, state_dim)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size, size=len(states))

        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            states_tensor = torch.from_numpy(states).float().to(self.device)  # Convert to tensor
            act_values = self.model(states_tensor)  # Forward pass
        return torch.argmax(act_values, dim=1).cpu().numpy()

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
        # q_values = self.model.predict(states, verbose=0)
        # q_next = self.target_model.predict(next_states, verbose=0)

        # Compute updated Q-values
        # keras/tf version
        # for i, (state, action, reward, next_state) in enumerate(minibatch):
        #     q_values[i][action] = reward + self.gamma * np.amax(q_next[i])

        # pytorch version
        with torch.no_grad():
            states_tensor = torch.from_numpy(states).float().to(self.device)
            next_states_tensor = torch.from_numpy(next_states).float().to(self.device)

            q_values_tensor = self.model(states_tensor)
            q_next = self.target_model(next_states_tensor)

        # Train in one batch
        # keras/tf version
        # self.model.fit(states, q_values, epochs=1, verbose=0)

        # pytorch version
        self.model.train()
        states_tensor = torch.from_numpy(states).float().to(self.device)
        # q_values_tensor = torch.from_numpy(q_values).float()
        predictions = self.model(states_tensor).to(self.device)
        loss = self.criterion(predictions, q_values_tensor).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # def update_target_model(self):
    #     self.target_model.set_weights(self.model.get_weights())

    # pytorch version
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())



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


def training(num_layers=1, neurons_per_layer=1, num_episode_batches=300, learning_rates=(.1, .01, .001),
             fill_memory_this_many_times=10, agent_memory_size=32 * 1024, set_epsilon_zero=False, logfile_stem='logfile', epsilon_at_halfpoint=.1,
             file_stem='file_stem'):
    start_time = time.time()
    prices = get_consumption_weight_curve(resample_in_minutes=60)
    env = SmartBuildingEnv(prices=prices.array)
    state_size = env.reset().shape[0]
    action_size = 3
    epsilon_decay = np.pow(epsilon_at_halfpoint, 2 / num_episode_batches)
    agent = DQNAgent(state_size, action_size, num_layers=num_layers, neurons_per_layer=neurons_per_layer,
                     epsilon_decay=epsilon_decay, memory_size=agent_memory_size)
    # agent.model.optimizer.learning_rate.assign(learning_rates[0])

    # fill memory of agent
    while True:
        state = env.reset()
        for _ in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state)
            state = next_state
            if done:
                break
        memory_fill_level = agent.memory.__sizeof__() / agent.memory.maxlen
        print(f'\rfilling agent memory... {memory_fill_level * 100 :.1f} %', end='', flush=True)
        if memory_fill_level >= 1:
            print('')
            break

    if set_epsilon_zero:  # for speed testing purposes - normally this should not be True
        agent.epsilon = 0
        agent.epsilon_min = 0

    # we want to fill the memory through training experiences fill_memory_this_many_times times
    # so we need to see how many episodes we need in a batch
    episodes_per_batch = int(np.floor(agent_memory_size * fill_memory_this_many_times / num_episode_batches / env.max_steps) + 1)
    env_list = [SmartBuildingEnv(prices=prices.array) for _ in range(episodes_per_batch)]
    act_array = np.zeros(episodes_per_batch)
    state_array = np.zeros((episodes_per_batch, state_size))
    next_state_array = np.zeros((episodes_per_batch, state_size))
    reward_array = np.zeros(episodes_per_batch)
    done_array = np.zeros(episodes_per_batch)
    avg_reward_array = np.zeros(num_episode_batches)
    done_array = np.zeros(episodes_per_batch, dtype=np.bool)

    log_keys = ['reward', 'episode_batch', 'iteration_time_in_s', 'epsilon', 'learning_rate']
    log_dict = dict()
    for key in log_keys:
        log_dict[key] = np.ones(num_episode_batches) * -1

    # logfile = open(logfile_stem + '.txt', 'w')
    logstring = f'starting training with {episodes_per_batch} episodes per batch'
    print(logstring)
    # logfile.write(logstring + '\n')
    global_start_time = time.time()

    # adjust_learning_rate_at = [int(num_episode_batches/3), int(num_episode_batches*2/3)]
    adjust_learning_rate_at = [int(num_episode_batches * (i + 1) / len(learning_rates))
                               for i in range(len(learning_rates) - 1)]

    for e in range(num_episode_batches):
        # if e in adjust_learning_rate_at:
        #     found_at = adjust_learning_rate_at.index(e)
        #     if found_at > 0:
        #         agent.model.optimizer.learning_rate.assign(learning_rates[found_at + 1])

        iteration_start_time = time.time()
        total_reward = 0
        # for env in env_list:
        # for i in range(episodes_per_batch):

        # state = env.reset()
        for i in range(episodes_per_batch):
            state_array[i, :] = env_list[i].reset()
            done_array[i] = False

        for _ in range(env.max_steps):
            act_array = agent.act_batch(state_array)
            for i in range(episodes_per_batch):
                if done_array[i]:
                    continue
                # action = agent.act(state) # this would be the thing to do according to Mnih et. al.
                next_state_array[i, :], reward_array[i], done_array[i] = env_list[i].step(act_array[i])
                agent.remember(state_array[i, :], act_array[i], reward_array[i], next_state_array[i, :])
                # state = next_state
                total_reward += reward_array[i]
            state_array = next_state_array.copy()

        agent.replay()
        agent.update_target_model()
        iteration_time = time.time() - iteration_start_time
        avg_reward = total_reward / episodes_per_batch
        # learning_rate = float(agent.model.optimizer.learning_rate)
        logstring = (f'Trainig... Episode batch: {e + 1}/{num_episode_batches}, Avg. reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.1e}, '
                     # f'learning rate: {learning_rate:.1e}, '
                     'iteration (s): {iteration_time:.2f}, '
                     f'duration (min): {(time.time() - global_start_time) / (e + 1) * num_episode_batches / 60.:.2f}, '
                     f'time left (min): {(time.time() - global_start_time) / (e + 1) * num_episode_batches / 60. - (time.time() - global_start_time) / 60:.2f}')

        log_dict['episode_batch'][e] = e
        log_dict['reward'][e] = avg_reward
        log_dict['iteration_time_in_s'][e] = iteration_time
        log_dict['epsilon'][e] = agent.epsilon
        # log_dict['learning_rate'][e] = learning_rate

        print(logstring, flush=True)
        # logfile.write(logstring + '\n')
        # plt.scatter(e, avg_reward_array[e])
        # plt.grid(True)
        # plt.show(block=True)
    logstring = f"Training complete. Time: {time.time() - start_time:.2f}. saving model to {file_stem + '.keras'}"
    print(logstring)
    # logfile.write(logstring + '\n')
    # logfile.close()
    df_log = pd.DataFrame(log_dict)
    # df_log.to_csv(logfile_stem + '.csv', index=False)
    df_log.to_csv(file_stem + '.csv', index=False)
    # agent.model.save(file_stem + '.keras')


# def test_model(file_model="chatgpt_deep_q_building.keras"):
# def test_model(axs, file_stem='chatgpt_deep_q_building'):
#     file_model = file_stem + '.keras'
#     df_log = pd.read_csv(file_stem + '.csv')
#
#     prices = get_consumption_weight_curve(resample_in_minutes=60)
#     env = SmartBuildingEnv(prices=prices.array)
#     state_size = env.reset().shape[0]
#     action_size = len(env.heating_levels)
#     agent = DQNAgent(state_size, action_size)
#     agent.model = keras.models.load_model(file_model)
#     agent.epsilon = 0
#     agent.epsilon_min = 0
#     rewards = np.zeros(24)
#     temperatures = np.zeros(24)
#     actions = np.zeros(24, dtype=int)
#
#     state = env.reset()
#     for i in range(24):
#         actions[i] = agent.act(state)
#         state, rewards[i], done = env.step(actions[i])
#         temperatures[i] = state[0]
#
#     # print(f"rewards over 24 hours: {sum(rewards[:24]):.2f}")
#     # fig, axs = plt.subplots(1, 3)
#
#     axs[0].plot(temperatures, label='Temperature')
#     axs[0].plot(actions, label='Heating level')
#     axs[0].plot(rewards, label='rewards')
#     axs[0].grid()
#     axs[0].legend()
#
#     axs[1].plot(env.daily_prices, label='energy price')
#     axs[1].plot(actions, label='Heating level')
#     axs[1].grid()
#     axs[1].legend()
#
#     axs[2].plot(df_log['reward'], color='grey', label='reward')
#     axs[2].plot(df_log['reward'].rolling(int(df_log.shape[0] / 10), center=True).mean(), label='reward_rolling', color='black', linewidth=1.5)
#     axs[2].plot(df_log['reward'].rolling(int(df_log.shape[0] / 10), center=True).mean(), label='reward_rolling', color='black', linewidth=1.5)
#     axs2 = axs[2].twinx()
#     axs2.plot(np.log10(df_log['learning_rate']), color='blue', label='log learning rate')
#     axs2.plot(np.log10(df_log['epsilon']), color='orange', label='log epsilon')
#     # axs2.set_ylim(0, 0.2)
#     axs2.legend()
#     axs[2].grid()
#     axs[2].legend()
#     # plt.show(block=True)


# def multitest(directory):
#     all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
#     all_stems = set([file[:file.find('.')] for file in all_files])
#
#     fig, axs = plt.subplots(nrows=max(len(all_stems), 2), ncols=3)
#     for i, stem in enumerate(all_stems):
#         axs[i, 0].set_title(stem)
#         test_model(axs[i, :], directory + '/' + stem)
#     plt.tight_layout()
#     plt.show(block=True)


def main():
    set_global_seed()

    path = './training_runs20250611_batch1'
    os.makedirs(path, exist_ok=True)
    runs = training_runs.get_runs20250611(save_to=path)
    for training_run in runs:
        training(**training_run)

    # test_model(file_stem='./training_dqn_building/dqn_building_n4x32_m76800')
    # multitest(path)


if __name__ == '__main__':
    main()
