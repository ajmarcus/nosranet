from annoy import AnnoyIndex
from typing import NamedTuple
from .config import Label, Model, FEATURES_FILE, SEED, SIZE, TEST_EXAMPLES, TREE
import json
from os import mkdir, path
import random
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor

random.seed(SEED)

PREPARE_PATH = "./nosranet/data"
PREPARE_TEMPLATE = "{path}/{model}_{label}.json"


class Dataset(NamedTuple):
    X_train: Tensor
    Y_train: Tensor
    X_dev: Tensor
    Y_dev: Tensor
    X_test: Tensor
    Y_test: Tensor


def split(X: Tensor, Y: Tensor) -> Dataset:
    assert X.shape[0] == Y.shape[0]

    all_examples = X.shape[0]
    train_examples = all_examples - TEST_EXAMPLES * 2
    return Dataset(
        X_train=X[:train_examples],
        Y_train=Y[:train_examples],
        X_dev=X[train_examples : train_examples + TEST_EXAMPLES],
        Y_dev=Y[train_examples : train_examples + TEST_EXAMPLES],
        X_test=X[train_examples + TEST_EXAMPLES :],
        Y_test=Y[train_examples + TEST_EXAMPLES :],
    )


def prepare(model: Model, label: Label) -> Dataset:
    if not path.exists(PREPARE_PATH):
        mkdir(PREPARE_PATH)

    features = []
    labels = []

    prepare_file = PREPARE_TEMPLATE.format(
        path=PREPARE_PATH, model=model.name, label=label.name
    )
    if path.isfile(prepare_file):
        with open(prepare_file, mode="r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                features.append(row["image"])
                labels.append(row["title"])
        return split(
            X=tf.convert_to_tensor(features),
            Y=tf.convert_to_tensor(labels),
        )
    else:
        examples = []
        title_tree = AnnoyIndex(SIZE[model], "angular")
        title_tree.load(TREE[model][label])

        with open(FEATURES_FILE, mode="r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                examples.append(
                    (
                        row["image"][model.name],
                        title_tree.get_item_vector(row["title_index"]),
                    )
                )
        random.shuffle(examples)

        with open(prepare_file, mode="w", encoding="utf-8") as f:
            for example in examples:
                feature, label = example
                features.append(feature)
                labels.append(label)
                row = {"image": feature, "title": label}
                f.write(f"{json.dumps(row)}\n")

        return split(
            X=tf.convert_to_tensor(features),
            Y=tf.convert_to_tensor(labels),
        )
