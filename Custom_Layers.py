import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pandas as pd

import itertools
from tqdm import tqdm

print(tf.__version__)

train_data, info = tfds.load('fashion_mnist', split="train", with_info=True, data_dir='./data/', download=True)
test_data = tfds.load('fashion_mnist', split='test', with_info=False, data_dir='./data/', download=False)

class_names = ["T-shirt/top", "Trouser/pants", "Pullover shirt", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
               "Ankle boot"]


def format_image(data):
    image = data['image']
    image = tf.cast(image, tf.float32) / 255.0
    return image, data['label']


train_data = train_data.map(format_image)
test_data = test_data.map(format_image)

batch_size = 64
train = train_data.shuffle(buffer_size=1024).batch(batch_size)
test = test_data.batch(batch_size)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, repetitions, kernel, pool_size=2):
        super().__init__()
        self.filters = filters
        self.repetitions = repetitions
        self.kernel = kernel
        self.maxpool = tf.keras.layers.MaxPooling2D()

    def build(self, input_shape):
        for i in range(self.repetitions):
            vars(self)[f'conv2D_{i}'] = tf.keras.layers.Conv2D(self.filters, self.kernel, input_shape=input_shape,
                                                               activation='relu', padding='same')
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=2)

    def call(self, inputs):
        x = self.conv2D_0(inputs)
        for i in range(1, self.repetitions):
            x = vars(self)[f'conv2D_{i}'](x)
        return self.maxpool(x)


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernels):
        super().__init__()
        self.filters = filters
        self.kernels = kernels

    def build(self, input_shape):
        self.batchnorm_1 = tf.keras.layers.BatchNormalization(input_shape=input_shape)
        self.conv2D_1 = tf.keras.layers.Conv2D(input_shape[-1], self.kernels, padding='same')
        self.batchnorm_2 = tf.keras.layers.BatchNormalization()
        self.conv2D_2 = tf.keras.layers.Conv2D(self.filters, self.kernels,  padding='same')
        self.conv2D_3 = tf.keras.layers.Conv2D(self.filters, 1, padding='same')

    def call(self, inputs):
        x = self.batchnorm_1(inputs)
        x = self.conv2D_1(x)
        x = tf.keras.activations.relu(x)
        x = self.batchnorm_2(x)
        x = self.conv2D_2(x)
        x = tf.keras.activations.relu(x)
        output = self.conv2D_3(inputs) + x
        return output


class TrainingModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.block_a = ConvBlock(64, 3, 3)
        self.resnetblock_a = ResNetBlock(128, 3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(128, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.block_a(inputs)
        x = self.resnetblock_a(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.classifier(x)


# test_model = tf.keras.Sequential([
#     tf.keras.Input(shape=(28,28,1)),
#     ConvBlock(64, 3, 3)
# ])
# test_model.summary()

@tf.function
def apply_gradient(model, optimizer, x, y, loss_object):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss_value = loss_object(y, y_pred)
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss_value, y_pred


def train_one_epoch(model, optimizer, train, loss_object):
    losses = []
    pbar = tqdm(total=len(list(enumerate(train))), position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')

    for x_batch, y_batch in train:
        loss_per_batch, y_pred = apply_gradient(model, optimizer, x_batch, y_batch, loss_object)
        losses.append(loss_per_batch)
        train_acc_metrics(y_batch, y_pred)
        pbar.update()
    return losses, train_acc_metrics


def perform_validataion(model, test, loss_object):
    losses = []
    for x_test_batch, y_test_batch in test:
        y_pred = model(x_test_batch)
        loss_test_batch = loss_object(y_test_batch, y_pred)
        losses.append(loss_test_batch)
        val_acc_metrics(y_test_batch, y_pred)
    return losses, val_acc_metrics


num_epochs = 1
epochs_val_losses, epochs_train_losses = [], []
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
model = TrainingModel(len(class_names))
import time
start_time = time.time()
for epoch in range(num_epochs):
    print(f"Start Epoch:{epoch} of {num_epochs} Epochs")
    losses_train, acc_train = train_one_epoch(model, optimizer, train, loss_object)

    losses_val, acc_val = perform_validataion(model, test, loss_object)

    epochs_train_losses.append(np.mean(losses_train))
    epochs_val_losses.append(np.mean(losses_val))

    print(
        f' Epoch {epoch}: Train Loss: {np.mean(losses_train):.3f}, Train Accuracy {acc_train.result().numpy():.2f} --- Validation Loss: {np.mean(losses_val):.3f}, Validation Accuracy {acc_val.result().numpy():.2f}')

print(f"----{time.time()-start_time:.2f} sec----")

#model = TrainingModel(len(class_names))
#model.build(input_shape = train.element_spec[0].shape)
#model.summary()

# earlystopping = tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_accuracy')
# model.compile(optimizer, loss=loss_object, metrics=['accuracy'])
# history = model.fit(train, validation_data=test, epochs=1, callbacks=[earlystopping])
# test = 1
