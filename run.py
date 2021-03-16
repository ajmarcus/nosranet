#!/usr/bin/env python

from annoy import AnnoyIndex
from datetime import datetime, time
from enum import Enum
import json
from typing import Tuple
import numpy as np
import random
from sys import argv
from tensorflow.keras import callbacks, layers, losses, metrics, optimizers, Sequential

# reproducible train/dev/test splits
random.seed(342908)

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
            titles.append(title_tree.get_item_vector(row["title_index"]))

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
    return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)


NUM_LAYERS = {
    "shallow": 1,
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
        (X_train, Y_train, X_dev, Y_dev, X_test, Y_test) = prepare(
            filename="./data/features/features.json",
            model=name,
            label=label,
        )
        model_layers = [layers.Flatten()]
        for l in range(NUM_LAYERS[argv[2]]):
            model_layers.append(layers.Dense(SIZE[name], activation="relu"))
            model_layers.append(layers.Dropout(DROPOUT_PROB))
        model_layers.append(layers.Dense(SIZE[name], activation="relu"))
        model = Sequential(model_layers)
        model.compile(
            loss=losses.cosine_similarity,
            metrics=[metrics.CosineSimilarity()],
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

# rn50 shallow
# Epoch 99/100
# 16/16 [==============================] - 2s 121ms/step - loss: -0.3991 - cosine_similarity: 0.3991 - val_loss: -0.3541 - val_cosine_similarity: 0.3541
# Epoch 100/100
# 16/16 [==============================] - 2s 123ms/step - loss: -0.3998 - cosine_similarity: 0.3998 - val_loss: -0.3537 - val_cosine_similarity: 0.3537
# 2/2 [==============================] - 0s 39ms/step - loss: -0.3498 - cosine_similarity: 0.3498

# rn50 deep
Epoch 33/100
# 16/16 [==============================] - 4s 241ms/step - loss: -0.3786 - cosine_similarity: 0.3786 - val_loss: -0.3621 - val_cosine_similarity: 0.3621