#!/usr/bin/env python

from annoy import AnnoyIndex
import clip
from enum import Enum
import json
import numpy as np
from PIL import Image
from os import mkdir, path
import torch
from typing import Any, Dict, List, Optional, Tuple

CUDA = "cuda"
CPU = "cpu"
VALID_DEVICE = set([CUDA, CPU])
CROP_PATH = "./data/crawl/crop"
DEVICE = CUDA if torch.cuda.is_available() else CPU
if DEVICE not in VALID_DEVICE:
    raise Exception(f"device: expected {VALID_DEVICE} actual {DEVICE}")
FEATURES_PATH = "./data/features"
TEMPLATE = "A photo of a {title}, a type of food."


class Model(Enum):
    rn50 = 1
    vit32 = 2


SIZE = {Model.rn50: 1024, Model.vit32: 512}
NAME = {Model.rn50: "RN50", Model.vit32: "ViT-B/32"}


class Clip(object):
    image: Dict[int, int] = {}
    title: Dict[str, int] = {}

    def __init__(self, model: Model):
        size = SIZE[model]
        name = model.name.lower()
        self.network, self.preprocess = clip.load(NAME[model], device=DEVICE)
        with open(f"./data/image/05_{name}.txt", mode="r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                self.image[int(line.strip().split("/")[4].split(".")[0])] = index
        self.feature = AnnoyIndex(size, "angular")
        self.feature.load(f"./data/image/05_{name}.ann")
        with open(
            f"./data/tree/05_{name}_01_join_False.txt", mode="r", encoding="utf-8"
        ) as f:
            for index, line in enumerate(f):
                self.title[line.strip()] = index
        self.label_plain = AnnoyIndex(size, "angular")
        self.label_plain.load(f"./data/tree/05_{name}_01_join_False.ann")
        self.label_context = AnnoyIndex(size, "angular")
        self.label_context.load(f"./data/tree/05_{name}_01_join_True.ann")

    def has_image(self, recipe_id: int) -> bool:
        return recipe_id in self.image.keys()

    def has_label(self, title: str) -> bool:
        return title in self.title.keys()

    def img2vec(self, recipe_id: int) -> np.ndarray:
        image = (
            self.preprocess(Image.open(f"{CROP_PATH}/{recipe_id}.jpeg"))
            .unsqueeze(0)
            .to(DEVICE)
        )
        with torch.no_grad():
            tensor = self.network.encode_image(image)
        return tensor.cpu().numpy()[0, :]

    def title2vec(self, title: str) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            tensor = self.network.encode_text(
                clip.tokenize([title, TEMPLATE.format(title=title)]).to(DEVICE)
            )
        matrix = tensor.cpu().numpy()
        title = matrix[0, :]
        context = matrix[1, :]
        return (title, context)

    def get_image_feature(self, recipe_id: int) -> List[float]:
        return self.feature.get_item_vector(self.image[recipe_id])

    def get_label_plain(self, title: str) -> List[float]:
        return self.label_plain.get_item_vector(self.title[title])

    def get_label_context(self, title: str) -> List[float]:
        return self.label_context.get_item_vector(self.title[title])

    def test_image(self, recipe_id: int) -> Optional[Any]:
        image = (
            self.preprocess(Image.open(f"{CROP_PATH}/{recipe_id}.jpeg"))
            .unsqueeze(0)
            .to(DEVICE)
        )
        with torch.no_grad():
            tensor = self.network.encode_image(image)
        expected = tensor.cpu().numpy()[0, :]
        actual = self.img2vec(recipe_id)
        if np.array_equal(expected, actual):
            return None
        else:
            return (expected, actual)

    def test_title(self, title: str) -> Optional[Any]:
        with torch.no_grad():
            tensor = self.network.encode_text(
                clip.tokenize([title, TEMPLATE.format(title=title)]).to(DEVICE)
            )
        matrix = tensor.cpu().numpy()
        expected_title = matrix[0, :]
        expected_context = matrix[1, :]
        actual_title, actual_context = self.title2vec(title)
        if np.array_equal(expected_title, actual_title) and np.array_equal(
            expected_context, actual_context
        ):
            return None
        else:
            if not np.array_equal(expected_title, actual_title) and not np.array_equal(
                expected_context, actual_context
            ):
                return (expected_title, actual_title, expected_context, actual_context)
            elif not np.array_equal(expected_title, actual_title):
                return (expected_title, actual_title, None, None)
            elif not np.array_equal(expected_context, actual_context):
                return (None, None, expected_context, actual_context)
        return None


if __name__ == "__main__":

    if not path.exists(FEATURES_PATH):
        mkdir(FEATURES_PATH)

    index = 0
    rn50 = Clip(model=Model.rn50)
    rn50_label = AnnoyIndex(SIZE[Model.rn50], "angular")
    rn50_context = AnnoyIndex(SIZE[Model.rn50], "angular")
    vit32 = Clip(model=Model.vit32)
    vit32_label = AnnoyIndex(SIZE[Model.vit32], "angular")
    vit32_context = AnnoyIndex(SIZE[Model.vit32], "angular")

    with open("./data/02_filter.json", mode="r", encoding="utf-8") as i, open(
        f"{FEATURES_PATH}/features.json", mode="w", encoding="utf-8"
    ) as o:
        for line in i:
            row = json.loads(line)
            rid = int(row["id"])
            title = row["title"]
            if (
                rn50.has_image(rid)
                and rn50.has_label(title)
                and vit32.has_image(rid)
                and vit32.has_label(title)
            ):
                row["image"] = {}
                row["title_index"] = index
                if DEVICE == CUDA:
                    row["image"]["rn50"] = rn50.img2vec(rid).tolist()
                    row["image"]["vit32"] = vit32.img2vec(rid).tolist()

                    rn50_title_label, rn50_title_context = rn50.title2vec(title)
                    rn50_label.add_item(index, rn50_title_label.tolist())
                    rn50_context.add_item(index, rn50_title_context.tolist())
                    vit32_title_label, vit32_title_context = vit32.title2vec(title)
                    vit32_label.add_item(index, vit32_title_label.tolist())
                    vit32_context.add_item(index, vit32_title_context.tolist())
                elif DEVICE == CPU:
                    row["image"]["rn50"] = rn50.get_image_feature(rid)
                    row["image"]["vit32"] = vit32.get_image_feature(rid)

                    rn50_label.add_item(index, rn50.get_label_plain(title))
                    rn50_context.add_item(index, rn50.get_label_context(title))
                    vit32_label.add_item(index, vit32.get_label_plain(title))
                    vit32_context.add_item(index, vit32.get_label_context(title))
                else:
                    raise Exception(f"device: expected {VALID_DEVICE} actual {DEVICE}")

                out = json.dumps(row)
                if index % 500 == 0:
                    print(f"wrote {index} lines")
                    row["image"]["rn50"] = len(row["image"]["rn50"])
                    row["image"]["vit32"] = len(row["image"]["vit32"])
                    print(json.dumps(row))
                if index % 10000 == 0 and DEVICE == CUDA:
                    rn50_test_image = rn50.test_image(rid)
                    rn50_test_title = rn50.test_title(title)
                    vit32_test_image = vit32.test_image(rid)
                    vit32_test_title = vit32.test_title(title)
                    if rn50_test_image is not None:
                        (expected, actual) = rn50_test_image
                        print(f"rn50 id: {rid}: expected {expected} actual {actual}")
                        raise Exception("")
                    if rn50_test_title is not None:
                        (
                            expected_title,
                            actual_title,
                            expected_context,
                            actual_context,
                        ) = rn50_test_title
                        print(
                            f"rn50 title: {title}: expected {expected_title} actual {actual_title}"
                        )
                        print(
                            f"rn50 context: {TEMPLATE.format(title=title)}: expected {expected_context} actual {actual_context}"
                        )
                        raise Exception("")
                    if vit32_test_image is not None:
                        (expected, actual) = vit32_test_image
                        print(f"vit32 id: {rid}: expected {expected} actual {actual}")
                        raise Exception("")
                    if vit32_test_title is not None:
                        (
                            expected_title,
                            actual_title,
                            expected_context,
                            actual_context,
                        ) = vit32_test_title
                        print(
                            f"vit32 title: {title}: expected {expected_title} actual {actual_title}"
                        )
                        print(
                            f"vit32 context: {TEMPLATE.format(title=title)}: expected {expected_context} actual {actual_context}"
                        )
                        raise Exception("")
                o.write(f"{out}\n")
                index += 1
    print(f"wrote {index} lines")
    print("build trees")
    rn50_label.build(1000)
    rn50_context.build(1000)
    vit32_label.build(1000)
    vit32_context.build(1000)
    print(
        f"tree size equal {rn50_label.get_n_items() == rn50_context.get_n_items() == vit32_label.get_n_items() == vit32_context.get_n_items()}"
    )
    print(f"tree size {rn50_label.get_n_items()}")
    print("save trees")
    rn50_label.save(f"{FEATURES_PATH}/rn50_label.ann")
    rn50_context.save(f"{FEATURES_PATH}/rn50_context.ann")
    vit32_label.save(f"{FEATURES_PATH}/vit32_label.ann")
    vit32_context.save(f"{FEATURES_PATH}/vit32_context.ann")
