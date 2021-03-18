from annoy import AnnoyIndex
from typing import List, NamedTuple, Tuple
from .config import Label, Model, Recipe, FEATURES_FILE, SEED, SIZE, TEST_EXAMPLES, TREE
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


class Split(NamedTuple):
    train_end: int
    test_end: int


class Testset(NamedTuple):
    X_test: Tensor
    Y_test: Tensor
    recipes: List[Recipe]


def calc_split(num_examples) -> Split:
    return Split(
        train_end=num_examples - TEST_EXAMPLES * 2,
        test_end=num_examples - TEST_EXAMPLES,
    )


def split(X: Tensor, Y: Tensor) -> Dataset:
    assert X.shape[0] == Y.shape[0]

    split = calc_split(X.shape[0])
    return Dataset(
        X_train=X[: split.train_end],
        Y_train=Y[: split.train_end],
        X_dev=X[split.train_end : split.test_end],
        Y_dev=Y[split.train_end : split.test_end],
        X_test=X[split.test_end :],
        Y_test=Y[split.test_end :],
    )


def prepare_test(model: Model, label: Label) -> Testset:
    num_examples = 0
    features = []
    labels = []
    recipes = []

    prepare_file = PREPARE_TEMPLATE.format(
        path=PREPARE_PATH, model=model.name, label=label.name
    )
    if not path.isfile(prepare_file):
        prepare(model=model, label=label)
    with open(prepare_file, mode="r", encoding="utf-8") as f:
        for line in f:
            num_examples += 0
            row = json.loads(line)
            features.append(row["feature"])
            labels.append(row["label"])
            recipes.append(
                Recipe(
                    id=row["id"],
                    title=row["title"],
                    title_index=row["title_index"],
                    ingredients=row["ingredients"],
                )
            )
    split = calc_split(num_examples=num_examples)
    return Testset(
        X_test=tf.convert_to_tensor(features[split.test_end :]),
        Y_test=tf.convert_to_tensor(labels[split.test_end :]),
        recipes=recipes[split.test_end :],
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
                features.append(row["feature"])
                labels.append(row["label"])
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
                        Recipe(
                            id=row["id"],
                            title=row["title"],
                            title_index=row["title_index"],
                            ingredients=row["ingredients"],
                        ),
                    )
                )
        random.shuffle(examples)

        with open(prepare_file, mode="w", encoding="utf-8") as f:
            for example in examples:
                feature, label, recipe = example
                features.append(feature)
                labels.append(label)
                row = {
                    "feature": feature,
                    "label": label,
                    "id": recipe.id,
                    "title": recipe.title,
                    "title_index": recipe.title_index,
                    "ingredients": recipe.ingredients,
                }
                f.write(f"{json.dumps(row)}\n")

        return split(
            X=tf.convert_to_tensor(features),
            Y=tf.convert_to_tensor(labels),
        )
