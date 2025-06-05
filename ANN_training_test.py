from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import pickle

class ModelInfo:
    def __init__(self, num_layers, neurons_per_layer):
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.precision = None


def setup_ann(input_dim, output_dim, num_layers, neurons_per_layer):
    model = keras.Sequential()
    # create model
    model.add(layers.Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(layers.Dense(neurons_per_layer, activation='relu'))
    model.add(layers.Dense(output_dim, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    # model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
    return model


def create_training_data(sample_size, output_dim, input_dim=1):
    states = np.empty((sample_size, input_dim))
    values = np.empty((sample_size, output_dim))
    very_low = -100
    low = -50
    mid = -25
    high = -10
    for i in range(sample_size):
        # create input
        states[i, 0] = random.uniform(10, 35)

        # compute output
        if states[i, 0] <= 15:
            values[i, :] = np.array([random.uniform(very_low, low), random.uniform(low, mid), random.uniform(mid, high)])
        if 15 < states[i, 0] <= 20:
            values[i, :] = np.array([random.uniform(very_low, low), random.uniform(mid, high), random.uniform(mid, high)])
        if 20 < states[i, 0] <= 24:
            values[i, :] = np.array([random.uniform(low, mid), random.uniform(low, mid), random.uniform(very_low, low)])
        if 24 < states[i, 0]:
            values[i, :] = np.array([random.uniform(mid, high), random.uniform(low, mid), random.uniform(very_low, low)])

    return {'states': states, 'values': values}


def test_model(model, test_sample_size=1000, plot=False):
    test_data = np.empty((test_sample_size, 1))
    test_numbers = np.random.uniform(10, 35, test_sample_size)
    test_data[:, 0] = test_numbers
    # results = np.empty((test_sample_size, output_dim))
    results = model.predict(test_data, verbose=0)

    num_correct = 0
    for i in range(len(test_numbers)):
        if (test_numbers[i] <= 15) and np.argmax(results[i]) == 2:
            num_correct += 1
        elif (15 < test_numbers[i] <= 20) and np.argmax(results[i]) == 1:
            num_correct += 1
        elif (20 < test_numbers[i] <= 24) and (np.argmax(results[i]) == 0 or np.argmax(results[i]) == 1):
            num_correct += 1
        elif (test_numbers[i] >= 24) and (np.argmax(results[i]) == 0):
            num_correct += 1

    if plot:
        plt.scatter(test_numbers, np.argmax(results, axis=1))
        plt.grid()
        plt.show(block=True)

    return num_correct / test_sample_size


def main(num_epochs=100, neuron_base=4):
    input_dim = 1
    output_dim = 3
    sample_size = 1000
    training_data = create_training_data(sample_size, output_dim)

    model_infos = []
    for num_epochs in [20, 40, 80, 160, 320]:
    # for num_epochs in [20]:
        for neuron_base in [1, 2, 4, 8]:
            print(f'\nnum_epochs: {num_epochs}, neuron_base: {neuron_base}')
            print('--------------------------------------------------------')

            nn_arch = ([(1, 64 * neuron_base), (2, 32 * neuron_base), (4, 16 * neuron_base), (8, 8 * neuron_base),
                        (16, 4 * neuron_base), (32, 2 * neuron_base), (64, 1 * neuron_base)])
            # model_infos = [ModelInfo(num_layers=arch[0], neurons_per_layer=arch[1]) for arch in nn_arch]
            # for model_info in model_infos:
            #     my_model = setup_ann(input_dim, output_dim, model_info.num_layers, model_info.neurons_per_layer)
            #     my_model.fit(training_data['states'], training_data['values'], epochs=num_epochs, verbose=0)
            #     model_info.precision = test_model(my_model)
            #     print(f'num_layers: {model_info.num_layers}, neurons_per_layer: {model_info.neurons_per_layer}, precision: {model_info.precision}')
            local_model_infos = [{'num_layers': arch[0], 'neurons_per_layer': arch[1]} for arch in nn_arch]
            for model_info in local_model_infos:
                my_model = setup_ann(input_dim, output_dim, **model_info)
                start_time = time.time()
                my_model.fit(training_data['states'], training_data['values'], epochs=num_epochs, verbose=0)
                model_info['training_time_per_epoch'] = (time.time() - start_time) / num_epochs
                model_info['accuracy'] = test_model(my_model)
                model_info['num_epochs'] = num_epochs
                model_info['neuron_base'] = neuron_base
                print(f'num_layers: {model_info["num_layers"]}, neurons_per_layer: {model_info["neurons_per_layer"]}, precision: {model_info["accuracy"]}')

    with open('model_infos.pkl', 'wb') as f:
        pickle.dump(model_infos, f)


if __name__ == '__main__':
    main()
    # for num_epochs in [20, 40, 80, 160, 320]:
    #     for neuron_base in [1, 2, 4, 8]:
    #         print(f'\nnum_epochs: {num_epochs}, neuron_base: {neuron_base}')
    #         print('--------------------------------------------------------')
    #         main(num_epochs=num_epochs, neuron_base=neuron_base)
