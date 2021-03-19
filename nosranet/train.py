from .config import Label, Model, MODEL_PATH, SEED, SIZE, TEST_EXAMPLES
from .prepare import prepare
import random
from tensorflow.keras import (
    activations,
    callbacks,
    layers,
    losses,
    metrics,
    optimizers,
    Sequential,
)
from os import path
from shutil import rmtree

# tf.config.run_functions_eagerly(True)
random.seed(SEED)

LOG_PATH = "./logs/{model}_{label}_layers:{num_layers}_dropout:{dropout_prob}"


def train(
    label: Label = Label.title,
    name: Model = Model.vit32,
    num_layers: int = 3,
    epochs: int = 5,
    dropout_prob: float = 0.4,
    batch_size: int = TEST_EXAMPLES,
):
    log_dir = LOG_PATH.format(
        model=name.name,
        label=label.name,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
    )
    model_dir = MODEL_PATH.format(
        model=name.name,
        label=label.name,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
    )
    if path.exists(log_dir):
        rmtree(log_dir, ignore_errors=True)
    if path.exists(model_dir):
        rmtree(model_dir, ignore_errors=True)

    data = prepare(model=name, label=label)

    model = Sequential()
    model.add(layers.Flatten())
    for l in range(num_layers):
        model.add(layers.Dense(SIZE[name]))
        model.add(layers.Activation(activations.relu))
        model.add(layers.Dropout(dropout_prob))
    model.add(layers.Dense(SIZE[name]))
    model.compile(
        loss=losses.cosine_similarity,
        metrics=[metrics.CosineSimilarity()],
        optimizer=optimizers.Adam(),
    )
    training = model.fit(
        x=data.X_train,
        y=data.Y_train,
        batch_size=batch_size,
        callbacks=[
            callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
            )
        ],
        epochs=epochs,
        validation_data=(data.X_dev, data.Y_dev),
    )
    results = model.evaluate(data.X_test, data.Y_test, batch_size=batch_size)
    model.save(model_dir)