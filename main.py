# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end],format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def trend(time, slope):
    return time * slope

def seasonal_pattern(season_time):
    return np.where(season_time < 0.1,
                    np.cos(season_time * 0.6 * np.pi),
                    2 * np.exp(9 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def plot_series(time, series, format="-", start=0, end=None):
    ax = plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)
    plt.gcf().legend("Values")

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(10 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.005
noise_level = 3

series = baseline + trend(time,slope) + seasonality(time, period=365, amplitude=amplitude) + noise(time, noise_level, seed=51)

split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

print("Your Output")
print(len(series))
plot_series(time, series)
plt.show()

def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x: x.batch(window_size+1))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).map(lambda x: (x[:-1], x[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)
print(dataset.element_spec)

inputs = tf.keras.Input([None, 1])
x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
x = tf.keras.layers.LSTM(32, return_sequences=False)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)
output = tf.keras.layers.Lambda(lambda x: x * 100)(x)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
EarlyStopping = tf.keras.callbacks.EarlyStopping(patience=5,monitor='mae')
model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9),
    loss=tf.keras.losses.Huber(),
    metrics=['mae'])
model.summary()
#img_file='/tmp/model_1.png'
#tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True)
history = model.fit(dataset, epochs=100, verbose=2, callbacks=[lr_schedule, EarlyStopping])
forecast = []

for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time+window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()