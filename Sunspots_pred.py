import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
import csv

time_step = []
temps = []
file_path = 'C:/Users/DEHACHA1/OneDrive - PG/Desktop/Mappe1.csv'
with open(file_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader)
    for l in reader:
        time_step.append(int(l[0]))
        temps.append(float(l[2]))

series = np.array(temps)
time = np.array(time_step)

# HYPERPARAMETERS
window_size = 20
batch_size = 250
validation_split = 0.1
time_split = int(len(series) * (1 - validation_split))
shuffle_buffer_size = 1000
num_epochs = 100

train_series = series[:time_split]
train_time = time[:time_split]
valid_series = series[time_split:]
valid_time = time[time_split:]


def data_preprocessing(series, window_size, batch_size, shuffle_buffer_size, training=True):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x: x.batch(window_size + 1))
    if training:
        dataset = dataset.shuffle(shuffle_buffer_size).map(lambda x: (x[:-1], x[-1:]))
        return dataset.batch(batch_size).prefetch(1)
    else:
        return dataset.map(lambda x: (x[:-1], x[-1:]))


def predict_series(model, series, window_size, start_predict=0, end_predict=None):
    forecast = []
    for time in np.arange(end_predict - start_predict):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
    results = np.array(forecast)[:, 0, 0]
    return results


dataset_train = data_preprocessing(train_series, window_size, batch_size, shuffle_buffer_size)
dataset_valid = data_preprocessing(valid_series, window_size, batch_size, shuffle_buffer_size)
tf.keras.backend.clear_session()
inputs = tf.keras.Input([None, 1])
temp = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='causal')(inputs)
temp = tf.keras.layers.MaxPooling1D(2)(temp)
temp = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(temp)
temp = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))(temp)
temp = tf.keras.layers.Dense(32, activation='relu')(temp)
temp = tf.keras.layers.Dense(10, activation='relu')(temp)
outputs = tf.keras.layers.Dense(1)(temp)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
lr_Scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch:  1e-8 * 10**(epoch/20))
early_Stopping = tf.keras.callbacks.EarlyStopping(patience=5, monitor='mae')
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9),
              metrics=['mae'])
history = model.fit(dataset_train, validation_data=dataset_valid, epochs=num_epochs, verbose=2, callbacks=[lr_Scheduler, early_Stopping])

# PLOTS
len_size = np.arange(len(history.history['loss']))
result = predict_series(model, series, window_size, time_split, len(series))
plt.plot(time[time_split:], result)
plt.plot(time[time_split:], series[time_split:])
plt.show()
plt.plot(len_size, history.history['val_mae'])
plt.plot(len_size, history.history['mae'])
plt.show()
