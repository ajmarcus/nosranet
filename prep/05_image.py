#!/usr/bin/env python

from annoy import AnnoyIndex
import clip
import json
import logging
import numpy as np
from PIL import Image
from os import listdir, mkdir, path
from time import time
from typing import List
import torch

# Build shared index for images

CROP_PATH = "./data/crawl/crop"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_TIME = int(time())
STEP_SIZE = 10000
MODEL_FILE = {"ViT-B/32": "vit32", "RN50": "rn50"}
TREE_PATH = "./data/image"


def log(message: str) -> str:
    hours = "{:.4f}".format((int(time()) - START_TIME) * 1.0 / (60 * 60))
    return f"{hours}h: {message}"


def get_image_filenames() -> List[str]:
    logging.info(log("start get image filenames"))
    return sorted(
        [
            path.join(CROP_PATH, f)
            for f in listdir(CROP_PATH)
            if path.isfile(path.join(CROP_PATH, f))
        ]
    )


def build_tree(image_filenames: List[str], model: str) -> bool:
    logging.info(log(f"{model} : start encode images"))
    network, preprocess = clip.load(model, device=DEVICE)
    tensors = []
    for index, filename in enumerate(image_filenames):
        if index % 100 == 0:
            logging.info(log(f"{model}: encoded {index} images"))
        image = preprocess(Image.open(filename)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            tensor = network.encode_image(image)
        tensors.append(tensor.cpu().numpy())
    matrix = np.concatenate(tensors, axis=0)
    logging.info(log(f"{model}: start load tree"))
    t = AnnoyIndex(matrix.shape[1], "angular")
    for i in range(matrix.shape[0]):
        if i % 5000 == 0:
            logging.info(log(f"{model}: load tree with {i} encodings"))
        t.add_item(i, matrix[i, :])
    t.build(1000)
    tree_filename = f"{TREE_PATH}/05_{MODEL_FILE[model]}.ann"
    logging.info(log(f"{model}: start save tree {tree_filename}"))
    t.save(tree_filename)
    logging.info(log(f"{model}: start save titles"))
    with open(
        f"{TREE_PATH}/05_{MODEL_FILE[model]}.txt",
        mode="w",
        encoding="utf-8",
    ) as f:
        for filenames in image_filenames:
            f.write(f"{filenames}\n")
    logging.info(log(f"{model}: end"))
    return True


if __name__ == "__main__":
    logging.basicConfig(
        filename=f"data/05_image.log",
        format="%(levelname)s:%(message)s",
        level=logging.INFO,
    )
    logging.info(log("start 05_image"))
    if not path.exists(TREE_PATH):
        mkdir(TREE_PATH)
    image_filenames = get_image_filenames()
    build_tree(image_filenames=image_filenames, model="ViT-B/32")
    build_tree(image_filenames=image_filenames, model="RN50")
    logging.info(log("end 05_image"))
