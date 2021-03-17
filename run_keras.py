#!/usr/bin/env python

from annoy import AnnoyIndex
from datetime import datetime, time
from enum import Enum
import json
from typing import Tuple
import numpy as np
import random
from sys import argv
import tensorflow as tf
from tensorflow.keras import callbacks, layers, losses, metrics, optimizers, Sequential
from tensorflow.python.framework.ops import Tensor

tf.config.run_functions_eagerly(True)
# reproducible train/dev/test splits
random.seed(342908)

DEBUG = False

FEATURES_PATH = "./data/features"
LOG_PATH = "./logs/review/{model}/{label}/{run}"
MODEL_PATH = "./model/review/{model}/{label}/{run}"

DROPOUT_PROB = 0.1
TEST_EXAMPLES = 8000
BATCH_SIZE = int(TEST_EXAMPLES / 2)


class Model(Enum):
    rn50 = 1
    vit32 = 2


MODEL_NAMES = set([m.name for m in Model])


class Label(Enum):
    context = 1
    title = 2


SIZE = {Model.rn50: 1024, Model.vit32: 512}
TREE = {
    Model.rn50: {
        Label.context: f"{FEATURES_PATH}/rn50_context.ann",
        Label.title: f"{FEATURES_PATH}/rn50_label.ann",
    },
    Model.vit32: {
        Label.context: f"{FEATURES_PATH}/vit32_context.ann",
        Label.title: f"{FEATURES_PATH}/vit32_label.ann",
    },
}


def prepare(filename: str, model: Model, label: Label = Label.title) -> Tuple:
    images = []
    titles = []

    title_tree = AnnoyIndex(SIZE[model], "angular")
    title_tree.load(TREE[model][label])

    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            images.append(row["image"][model.name])
            titles.append(row["title_index"])

    num_classes = len(set(titles))
    X = np.asarray(images)
    Y = np.asarray(titles)
    del images
    del titles

    # shuffle X & Y
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    X = X[indexes]
    Y = Y[indexes]

    all_examples = X.shape[0]
    train_examples = all_examples - TEST_EXAMPLES * 2
    X_train = X[:train_examples]
    X_dev = X[train_examples : train_examples + TEST_EXAMPLES]
    X_test = X[train_examples + TEST_EXAMPLES :]
    del X

    Y_train = Y[:train_examples]
    Y_dev = Y[train_examples : train_examples + TEST_EXAMPLES]
    Y_test = Y[train_examples + TEST_EXAMPLES :]
    del Y
    return (num_classes, X_train, Y_train, X_dev, Y_dev, X_test, Y_test)


class KNN(object):
    def __init__(self, num_classes: int, label: Label, model: Model):
        self.num_classes = num_classes
        self.tree = AnnoyIndex(SIZE[model], "angular")
        self.tree.load(TREE[model][label])

    def get_nearest_one(self, image_vector: Tensor) -> Tensor:
        title_index = self.tree.get_nns_by_vector(vector=image_vector.numpy(), n=1)[0]
        return tf.one_hot(indices=[title_index], depth=self.num_classes)

    def get_nearest_batch(self, batch: Tensor):
        return tf.map_fn(self.get_nearest_one, elems=batch)

    def calc_loss(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        one_hot_pred = self.get_nearest_batch(batch=y_pred)
        return losses.sparse_categorical_crossentropy(
            y_true=y_true, y_pred=one_hot_pred
        )


NUM_LAYERS = {
    "shallow": 0,
    "deep": 3,
}

if __name__ == "__main__":
    if len(argv) != 3 or argv[1] not in MODEL_NAMES or argv[2] not in NUM_LAYERS.keys():
        name_arg = "|".join(list(MODEL_NAMES))
        depth_arg = "|".join(NUM_LAYERS.keys())
        print(f"usage: ./run.py {name_arg} {depth_arg}")
    else:
        name = Model[argv[1]]
        label = Label.title
        (num_classes, X_train, Y_train, X_dev, Y_dev, X_test, Y_test) = prepare(
            filename="./data/features/features.json",
            model=name,
            label=label,
        )
        knn = KNN(num_classes=num_classes, label=label, model=name)

        model = Sequential()
        model.add(layers.Flatten())
        for l in range(NUM_LAYERS[argv[2]]):
            model.add(layers.Dense(SIZE[name], activation="relu"))
            model.add(layers.Dropout(DROPOUT_PROB))
        model.add(layers.Dense(SIZE[name]))
        model.compile(
            loss=knn.calc_loss,
            metrics=[
                metrics.SparseCategoricalAccuracy(),
            ],
            optimizer=optimizers.Adam(),
        )
        run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        training = model.fit(
            x=X_train,
            y=Y_train,
            batch_size=BATCH_SIZE,
            callbacks=[
                callbacks.TensorBoard(
                    log_dir=LOG_PATH.format(model=name, label=label, run=run_time),
                    histogram_freq=1,
                )
            ],
            epochs=5,
            validation_data=(X_dev, Y_dev),
        )
        results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
        model.save(MODEL_PATH.format(model=name, label=label, run=run_time))