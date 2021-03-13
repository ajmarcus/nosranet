#!/usr/bin/env python

from annoy import AnnoyIndex
import clip
import json
import logging
import numpy as np
from time import time
import torch

# Build shared index for images and recipe titles

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_TIME = int(time())
MODEL_FILE = {"ViT-B/32": "vit32", "RN50": "rn50"}


def log(message: str) -> str:
    hours = "{:.4f}".format((int(time()) - START_TIME) * 1.0 / (60 * 60))
    return f"{hours}h: {message}"


def build_tree(model: str, recipe: str) -> bool:
    logging.info(log(f"{model} {recipe}: start read titles"))
    titles = set([])
    with open(f"./data/{recipe}.json", mode="r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            titles.add(row["title"])
    titles = sorted(list(titles))
    midpoint = round(len(titles) / 2.0)
    logging.info(log(f"{model} {recipe}: read {len(titles)} titles"))
    logging.info(log(f"{model} {recipe}: start encode titles"))
    network, preprocess = clip.load(model, device=DEVICE)
    text = clip.tokenize(titles).to(DEVICE)
    with torch.no_grad():
        tensor = network.encode_text(text)
    matrix = tensor.cpu().numpy()
    logging.info(log(f"{model} {recipe}: start load tree"))
    t = AnnoyIndex(matrix.shape[1], "angular")
    for i in range(matrix.shape[0]):
        if i % 5000 == 0:
            logging.info(log(f"{model} {recipe}: load tree with {i} encodings"))
        t.add_item(i, matrix[i, :])
    t.build(1000)
    tree_filename = f"./data/05_{MODEL_FILE[model]}_{recipe}.ann"
    logging.info(log(f"{model} {recipe}: start save tree {tree_filename}"))
    t.save(tree_filename)
    logging.info(log(f"{model} {recipe}: start save titles"))
    with open(
        f"./data/05_{MODEL_FILE[model]}_{recipe}.txt", mode="w", encoding="utf-8"
    ) as f:
        for t in titles:
            f.write(f"{t}\n")
    logging.info(log(f"{model} {recipe}: end"))
    return True


if __name__ == "__main__":
    logging.basicConfig(
        filename=f"data/05_annoy.log",
        format="%(levelname)s:%(message)s",
        level=logging.INFO,
    )
    logging.info(log("start 05_annoy"))
    build_tree(model="ViT-B/32", recipe="04_crop")
    build_tree(model="RN50", recipe="04_crop")
    logging.info(log("end 05_annoy"))
