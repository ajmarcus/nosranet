#!/usr/bin/env python

# Split train/dev/test datasets
# Read chapter on "Deep learning for text and sequences" in Deep Learning with Python by Francois Chollet
# Article "Do Good Recipes Need Butter? Predicting user ratings of online recipes"
# https://scholar.google.com/scholar?cites=8686827231755067221&as_sdt=5,32&sciodt=0,32&hl=en

# count_review: Counter({5: 61219, 3: 6420, 2: 4605, 1: 298, 0: 100})
# recipes: 72642
# reviews: 526742
# max_ingredients: 11
# min_ingredients: 1
# max_reviews: 0
# min_reviews: 2
# num_negative: 11423
# num_positive: 61219
# all_ingredients_size: 15575

from datetime import datetime
import json
from typing import Tuple
import numpy as np
import random
from sys import argv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import callbacks, layers, losses, metrics, optimizers, Sequential

# reproducible train/dev/test splits
random.seed(2308923)

ALL_EXAMPLES = 72642
DROPOUT_PROB = 0.6
MAX_INGREDIENTS = 4096
TEST_EXAMPLES = 10000
TRAIN_EXAMPLES = ALL_EXAMPLES - TEST_EXAMPLES * 2
BATCH_SIZE = int(TEST_EXAMPLES / 2)
LOG_DIR = "./logs/review/{network}/{run}"
MODEL_DIR = "./model/review/{network}/{run}"


def prepare(filename: str, max_ingredients: int) -> Tuple:
    ingredients = []
    labels = []

    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ingredients.append(row["ingredients"])
            labels.append(row["label"])

    tokenizer = Tokenizer(num_words=max_ingredients)
    tokenizer.fit_on_texts(ingredients)
    # one hot encoding
    X = tokenizer.texts_to_matrix(ingredients, mode="binary")
    Y = np.asarray(labels)

    del tokenizer
    del ingredients
    del labels

    # shuffle X & Y
    examples = np.arange(X.shape[0])
    np.random.shuffle(examples)
    X = X[examples]
    Y = Y[examples]

    X_train = X[:TRAIN_EXAMPLES]
    X_dev = X[TRAIN_EXAMPLES : TRAIN_EXAMPLES + TEST_EXAMPLES]
    X_test = X[TRAIN_EXAMPLES + TEST_EXAMPLES :]
    del X

    Y_train = Y[:TRAIN_EXAMPLES]
    Y_dev = Y[TRAIN_EXAMPLES : TRAIN_EXAMPLES + TEST_EXAMPLES]
    Y_test = Y[TRAIN_EXAMPLES + TEST_EXAMPLES :]
    del Y
    return (X_train, Y_train, X_dev, Y_dev, X_test, Y_test)


networks = {
    "shallow": {
        "max_ingredients": 8192,
        "layers": [
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(DROPOUT_PROB),
            layers.Dense(1, activation="sigmoid"),
        ],
    },
    "deep": {
        "max_ingredients": 8192,
        "layers": [
            layers.Flatten(),
            layers.Dense(2048, activation="relu"),
            layers.Dropout(DROPOUT_PROB),
            layers.Dense(512, activation="relu"),
            layers.Dropout(DROPOUT_PROB),
            layers.Dense(128, activation="relu"),
            layers.Dropout(DROPOUT_PROB),
            layers.Dense(1, activation="sigmoid"),
        ],
    },
}

if __name__ == "__main__":
    if len(argv) != 2 or argv[1] not in networks.keys():
        options = "|".join(networks.keys())
        print(f"usage: ./predict_review.py {options}")
    else:
        name = argv[1]
        (X_train, Y_train, X_dev, Y_dev, X_test, Y_test) = prepare(
            filename="./data/02_filter.json",
            max_ingredients=int(networks[name]["max_ingredients"]),
        )
        model = Sequential(networks[name]["layers"])
        model.compile(
            loss=losses.binary_crossentropy,
            metrics=[metrics.AUC()],
            optimizer=optimizers.Adam(),
        )
        run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        training = model.fit(
            x=X_train,
            y=Y_train,
            batch_size=BATCH_SIZE,
            callbacks=[
                callbacks.TensorBoard(
                    log_dir=LOG_DIR.format(network=name, run=run_time),
                    histogram_freq=1,
                )
            ],
            epochs=5,
            validation_data=(X_dev, Y_dev),
        )
        results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
        model.save(MODEL_DIR.format(network=name, run=run_time))

# ./predict_review.py shallow
# 11/11 [==============================] - 3s 174ms/step - loss: 0.6504 - auc: 0.4921 - val_loss: 0.5214 - val_auc: 0.4795
# Epoch 2/5
# 11/11 [==============================] - 1s 83ms/step - loss: 0.5027 - auc: 0.5076 - val_loss: 0.4586 - val_auc: 0.4849
# Epoch 3/5
# 11/11 [==============================] - 1s 81ms/step - loss: 0.4616 - auc: 0.5245 - val_loss: 0.4505 - val_auc: 0.4998
# Epoch 4/5
# 11/11 [==============================] - 1s 82ms/step - loss: 0.4552 - auc: 0.5346 - val_loss: 0.4429 - val_auc: 0.5221
# Epoch 5/5
# 11/11 [==============================] - 1s 71ms/step - loss: 0.4452 - auc: 0.5698 - val_loss: 0.4370 - val_auc: 0.5466
# 2/2 [==============================] - 0s 29ms/step - loss: 0.4474 - auc: 0.5651

# ./predict_review.py deep
# 11/11 [==============================] - 10s 803ms/step - loss: 0.5450 - auc: 0.5043 - val_loss: 0.4555 - val_auc: 0.5808
# Epoch 2/5
# 11/11 [==============================] - 8s 719ms/step - loss: 0.4451 - auc: 0.5891 - val_loss: 0.4266 - val_auc: 0.6103
# Epoch 3/5
# 11/11 [==============================] - 8s 737ms/step - loss: 0.4189 - auc: 0.6422 - val_loss: 0.4252 - val_auc: 0.6173
# Epoch 4/5
# 11/11 [==============================] - 8s 737ms/step - loss: 0.4103 - auc: 0.6755 - val_loss: 0.4261 - val_auc: 0.6185
# Epoch 5/5
# 11/11 [==============================] - 8s 733ms/step - loss: 0.3959 - auc: 0.7153 - val_loss: 0.4293 - val_auc: 0.6156
# 2/2 [==============================] - 1s 298ms/step - loss: 0.4317 - auc: 0.6012