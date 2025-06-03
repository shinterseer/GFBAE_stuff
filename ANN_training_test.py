from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
import matplotlib.pyplot as plt


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


def main():
    # setup ann
    input_dim = 1
    output_dim = 3
    num_layers = 16
    neurons_per_layer = 32
    my_model = setup_ann(input_dim, output_dim, num_layers, neurons_per_layer)

    # create training data
    # low temperatures give high actions, we could use the q-learner table as training? no. just manually
    sample_size = 1000
    very_low = -100
    low = -50
    mid = -25
    high = -10

    states = np.empty((sample_size, 1))
    values = np.empty((sample_size, output_dim))
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

    # train ann
    my_model.fit(states, values, epochs=50, verbose=1)

    # test
    test_sample_size = 1000
    test_data = np.empty((test_sample_size, 1))
    test_numbers = np.random.uniform(10, 35, test_sample_size)
    test_data[:, 0] = test_numbers
    # results = np.empty((test_sample_size, output_dim))
    results = my_model.predict(test_data, verbose=0)

    plt.scatter(test_numbers, np.argmax(results, axis=1))
    plt.grid()
    plt.show(block=True)

    x=0

    pass

if __name__ == '__main__':
    main()
