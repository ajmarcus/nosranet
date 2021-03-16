#!/usr/bin/env python

# Resources:
# Chapter on "Deep learning for computer vision" in Deep Learning with Python by Francois Chollet
# https://www.tensorflow.org/datasets/keras_example
# https://www.tensorflow.org/tutorials/images/classification
# https://stackoverflow.com/questions/46135499/how-to-properly-combine-tensorflows-dataset-api-and-keras
# https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/fashion_mnist.ipynb

# Prior research
# https://arxiv.org/abs/1810.06553
# https://github.com/torralba-lab/im2recipe-Pytorch
# http://wednesday.csail.mit.edu/pretrained/model_e220_v-4.700.pth.tar

import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, models, optimizers
from tensorflow.python.keras.layers.convolutional import Conv2D
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1024
IMAGE_SIZE = 32

(train, validation) = tfds.load(
    name="food101",
    split=["train", "validation"],
    shuffle_files=True,
    as_supervised=True,
)


def resize(image, label):
    image = tf.image.resize_with_crop_or_pad(
        image, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE
    )
    return image, label


train = (
    train.map(resize, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(buffer_size=BATCH_SIZE * 4)
    .batch(batch_size=BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)
validation = (
    validation.map(resize, num_parallel_calls=AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

model = models.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.Conv2D(
            filters=32,
            kernel_size=(4, 4),
            activation="relu",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE),
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dense(101, activation=None),
    ]
)
model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[metrics.SparseCategoricalAccuracy()],
)
model.fit(
    train,
    epochs=5,
    validation_data=validation,
)

# Epoch 1/5
# 2021-02-28 12:40:56.219529: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
# 74/74 [==============================] - 41s 529ms/step - loss: 4.5479 - sparse_categorical_accuracy: 0.0258 - val_loss: 4.3710 - val_sparse_categorical_accuracy: 0.0483
# Epoch 2/5
# 74/74 [==============================] - 15s 201ms/step - loss: 4.3756 - sparse_categorical_accuracy: 0.0495 - val_loss: 4.3269 - val_sparse_categorical_accuracy: 0.0564
# Epoch 3/5
# 74/74 [==============================] - 15s 201ms/step - loss: 4.3163 - sparse_categorical_accuracy: 0.0579 - val_loss: 4.2965 - val_sparse_categorical_accuracy: 0.0608
# Epoch 4/5
# 74/74 [==============================] - 15s 201ms/step - loss: 4.2657 - sparse_categorical_accuracy: 0.0664 - val_loss: 4.2684 - val_sparse_categorical_accuracy: 0.0654
# Epoch 5/5
# 74/74 [==============================] - 15s 201ms/step - loss: 4.2128 - sparse_categorical_accuracy: 0.0738 - val_loss: 4.2544 - val_sparse_categorical_accuracy: 0.0659