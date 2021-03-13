#!/usr/bin/env python

from annoy import AnnoyIndex
import clip
import json
import logging
import numpy as np
from os import mkdir, path
from time import time
import torch

# Build shared index for recipe titles

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_TIME = int(time())
STEP_SIZE = 10000
MODEL_FILE = {"ViT-B/32": "vit32", "RN50": "rn50"}
TEMPLATE = "A photo of a {title}, a type of food."
TREE_PATH = "./data/tree"


def log(message: str) -> str:
    hours = "{:.4f}".format((int(time()) - START_TIME) * 1.0 / (60 * 60))
    return f"{hours}h: {message}"


def encode_text(network, titles):
    text = clip.tokenize(titles).to(DEVICE)
    with torch.no_grad():
        tensor = network.encode_text(text)
    return tensor.cpu().numpy()


def format_template(title: str) -> str:
    return TEMPLATE.format(title=title)


def build_tree(model: str, recipe: str, template: bool) -> bool:
    logging.info(log(f"{model} {recipe}: start read titles"))
    titles_set = set([])
    with open(f"./data/{recipe}.json", mode="r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            titles_set.add(row["title"])
    titles = sorted(list(titles_set))
    if template:
        titles = list(map(format_template, titles))
    logging.info(log(f"{model} {recipe}: read {len(titles)} titles"))
    logging.info(log(f"{model} {recipe}: start encode titles"))
    network, preprocess = clip.load(model, device=DEVICE)
    tensors = []
    for i in range(round(len(titles) / STEP_SIZE)):
        start = i * STEP_SIZE
        end = min([len(titles), (i + 1) * STEP_SIZE])
        logging.info(log(f"{model} {recipe}: encode titles from {start} to {end}"))
        tensors.append(encode_text(network=network, titles=titles[start:end]))
    matrix = np.concatenate(tensors, axis=0)
    logging.info(log(f"{model} {recipe}: start load tree"))
    t = AnnoyIndex(matrix.shape[1], "angular")
    for i in range(matrix.shape[0]):
        if i % 5000 == 0:
            logging.info(log(f"{model} {recipe}: load tree with {i} encodings"))
        t.add_item(i, matrix[i, :])
    t.build(1000)
    tree_filename = f"{TREE_PATH}/05_{MODEL_FILE[model]}_{recipe}_{template}.ann"
    logging.info(log(f"{model} {recipe}: start save tree {tree_filename}"))
    t.save(tree_filename)
    logging.info(log(f"{model} {recipe}: start save titles"))
    with open(
        f"{TREE_PATH}/05_{MODEL_FILE[model]}_{recipe}_{template}.txt",
        mode="w",
        encoding="utf-8",
    ) as f:
        for t in titles:
            f.write(f"{t}\n")
    logging.info(log(f"{model} {recipe}: end"))
    return True


if __name__ == "__main__":
    logging.basicConfig(
        filename=f"data/05_tree.log",
        format="%(levelname)s:%(message)s",
        level=logging.INFO,
    )
    logging.info(log("start 05_tree"))
    if not path.exists(TREE_PATH):
        mkdir(TREE_PATH)
    build_tree(model="ViT-B/32", recipe="01_join", template=False)
    build_tree(model="ViT-B/32", recipe="01_join", template=True)
    build_tree(model="RN50", recipe="01_join", template=False)
    build_tree(model="RN50", recipe="01_join", template=True)
    build_tree(model="ViT-B/32", recipe="04_crop", template=False)
    build_tree(model="ViT-B/32", recipe="04_crop", template=True)
    build_tree(model="RN50", recipe="04_crop", template=False)
    build_tree(model="RN50", recipe="04_crop", template=True)
    logging.info(log("end 05_tree"))
