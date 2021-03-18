#!/usr/bin/env python

from typing import NamedTuple
from annoy import AnnoyIndex
from .config import Label, FEATURES_FILE, MODEL_PATH, Model, SIZE, TREE, log
from .prepare import prepare_test
from .train import train
import json
import numpy as np
from os import path
import tensorflow as tf


class Metrics(NamedTuple):
    accuracy: float  # correct / total


class KNN(object):
    def __init__(self, label: Label, model: Model):
        self.tree = AnnoyIndex(SIZE[model], "angular")
        self.tree.load(TREE[model][label])

    def nearest_index(self, y_pred: np.ndarray) -> int:
        return self.tree.get_nns_by_vector(vector=y_pred.tolist(), n=1)[0]

    def nearest(self, y_pred: np.ndarray) -> np.ndarray:
        index = self.nearest_index(y_pred=y_pred)
        return np.asarray(self.tree.get_item_vector(index))

    def distance(self, left_index: int, right_index: int) -> float:
        return self.tree.get_distance(left_index, right_index)


def calc_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray):
    assert Y_true.shape[0] == Y_pred.shape[0]
    num_correct = np.sum(Y_true == Y_pred)
    return num_correct * 1.0 / Y_true.shape[0]


def evaluate_baseline(
    label: Label = Label.title,
    name: Model = Model.vit32,
) -> float:
    knn = KNN(label=label, model=name)
    data = prepare_test(model=name, label=label)
    Y_true = np.asarray([r.title_index for r in data.recipes])
    X_nearest = np.apply_along_axis(knn.nearest_index, axis=1, arr=data.X_test.numpy())
    accuracy = calc_accuracy(Y_true=Y_true, Y_pred=X_nearest)
    print("========================================================")
    print("========================================================")
    print(f"({name.name},{label.name}) baseline accuracy: {accuracy}")
    print("========================================================")
    print("========================================================")
    return accuracy


def evaluate(
    label: Label = Label.title,
    name: Model = Model.vit32,
    num_layers: int = 3,
    dropout_prob: float = 0.1,
    epochs: int = 5,
):
    knn = KNN(label=label, model=name)
    data = prepare_test(model=name, label=label)
    Y_true = np.asarray([r.title_index for r in data.recipes])

    model_dir = MODEL_PATH.format(
        model=name.name,
        label=label.name,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
    )
    if not path.exists(model_dir):
        train(
            label=label,
            name=name,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            epochs=epochs,
        )
    model = tf.keras.models.load_model(model_dir)
    Y_pred = model.predict(data.X_test)
    Y_nearest = np.apply_along_axis(knn.nearest_index, axis=1, arr=Y_pred)
    accuracy = calc_accuracy(Y_true=Y_true, Y_pred=Y_nearest)
    print("========================================================")
    print("========================================================")
    print(
        f"({name.name},{label.name},layers:{num_layers},dropout:{dropout_prob},epochs:{epochs}) accuracy: {accuracy}"
    )
    print("========================================================")
    print("========================================================")
    return accuracy
