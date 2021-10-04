import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random


def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs.append([x[z1], x[z2]])
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs.append([x[z1], x[z2]])
            labels += [1, 0]
    return np.array(pairs, dtype="float32"), np.array(labels)


def create_pairs_on_set(images, labels):
    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype('float32')

    return pairs, y


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")
train_images = train_images / 255.
test_images = test_images / 255.

tr_pairs, tr_y = create_pairs_on_set(train_images, train_labels)
ts_pairs, ts_y = create_pairs_on_set(test_images, test_labels)

this_pair = 8


# show_image(tr_pairs[this_pair][0])
# show_image(tr_pairs[this_pair][1])
# print(tr_y[this_pair])


def initialize_base_network():
    inputs = Input(shape=(28, 28), name="Base")
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    output = Dense(128, activation='relu')(x)

    return Model(inputs=inputs, outputs=output)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


base_network = initialize_base_network()

input_a = Input(shape=(28, 28))
vect_output_a = base_network(input_a)
input_b = Input(shape=(28, 28))
vect_output_b = base_network(input_b)

output = Lambda(euclidean_distance)([vect_output_a, vect_output_b])

model = Model(inputs=[input_a, input_b], outputs=output)
model.summary()


# plot_model(base_network, show_shapes=True, show_layer_names=True, to_file='base-model.png')

# def contrastive_loss_with_margin(margin):
#     def contrastive_loss(y_true, y_pred):
#         '''Contrastive loss from Hadsell-et-al.'06
#         http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#         '''
#         square_pred = K.square(y_pred)
#         margin_square = K.square(K.maximum(margin - y_pred, 0))
#         return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
#     return contrastive_loss

class ContractiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


rms = RMSprop()
model.compile(loss=ContractiveLoss(margin=1), optimizer=rms)
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, epochs=20, batch_size=128,
                    validation_data=([ts_pairs[:, 0], ts_pairs[:, 1]], ts_y))
