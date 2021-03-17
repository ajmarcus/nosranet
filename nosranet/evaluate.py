#!/usr/bin/env python

from typing import Dict
from annoy import AnnoyIndex
from .config import Label, FEATURES_FILE, MODEL_PATH, Model, SIZE, TREE
from .prepare import prepare
from .train import train
import json
import numpy as np
from os import path
import tensorflow as tf


class ImageLookup(object):
    def __init__(self, model: Model) -> None:
        self.vector_id = {}
        with open(FEATURES_FILE, mode="r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.vector_id[np.asarray(row["image"][model.name])] = int(row["id"])

    def get(self, vector: tf.Tensor) -> int:
        return self.vector_id[vector.numpy()]


class TitleLookup(object):
    def __init__(self) -> None:
        self.index_title = {}
        with open(FEATURES_FILE, mode="r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.index_title[int(row["title_index"])] = {
                    "id": int(row["id"]),
                    "title": row["title"],
                }

    def get(self, index: int) -> Dict:
        return self.index_title[index]


class KNN(object):
    def __init__(self, label: Label, model: Model):
        self.tree = AnnoyIndex(SIZE[model], "angular")
        self.tree.load(TREE[model][label])

    def nearest(self, y_pred: tf.Tensor) -> tf.Tensor:
        index = self.tree.get_nns_by_vector(vector=y_pred.numpy(), n=1)[0]
        return tf.convert_to_tensor(self.tree.get_item_vector(index))

    def distance(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        return self.tree.get_distance(y_true.numpy(), y_pred.numpy())


def evaluate(
    label: Label = Label.title,
    name: Model = Model.vit32,
    num_layers: int = 3,
):
    model_dir = MODEL_PATH.format(
        model=name.name, label=label.name, num_layers=num_layers
    )
    if not path.exists(model_dir):
        train(label=label, name=name, num_layers=num_layers)
    knn = KNN(label=label, model=name)
    model = tf.keras.models.load_model(model_dir)
    data = prepare(model=name, label=label)
    Y_pred = model.predict(data.X_test[:1])
    Y_nearest = tf.map_fn(knn.nearest, Y_pred)
    print(tf.math.reduce_sum(Y_pred))
    print(tf.math.reduce_sum(Y_nearest))
    results = tf.math.equal(data.Y_test[:1], Y_nearest)
    print(results[:10])
