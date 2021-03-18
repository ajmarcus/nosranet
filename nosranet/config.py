from enum import Enum
from time import time
from typing import List, NamedTuple


class Model(Enum):
    rn50 = 1
    vit32 = 2


class Label(Enum):
    context = 1
    title = 2


class Recipe(NamedTuple):
    id: int
    title: str
    title_index: int
    ingredients: List[str]


CROP_PATH = "./data/crawl/crop"
FEATURES_PATH = "./data/features"
FEATURES_FILE = "./data/features/features.json"
MODEL_PATH = "./model/{model}_{label}_layers:{num_layers}_dropout:{dropout_prob}"

TEST_EXAMPLES = 8000

SEED = 342908
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

START_TIME = int(time())


def log(message: str) -> str:
    hours = "{:.4f}".format((int(time()) - START_TIME) * 1.0 / (60 * 60))
    return f"{hours}h: {message}"