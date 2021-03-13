#!/usr/bin/env python

from os import listdir, mkdir, path
import json
from typing import Set
from PIL import Image, ExifTags

CROP_PATH = "./data/crawl/crop"
IMAGE_PATH = "./data/crawl/image/"
IMAGE_ENDINGS = set(["jpeg"])
IMAGE_RESOLUTION = 224  # same as what was used in the CLIP paper

EXIF_ORIENTATION = [
    tag for tag, name in ExifTags.TAGS.items() if name == "Orientation"
][0]

ROTATE = [
    None,  # 0: no exif
    None,  # 1: no rotation
    Image.FLIP_LEFT_RIGHT,  # 2
    Image.ROTATE_180,  # 3
    Image.FLIP_TOP_BOTTOM,  # 4
    Image.TRANSPOSE,  # 5
    Image.ROTATE_270,  # 6
    Image.TRANSVERSE,  # 7
    Image.ROTATE_90,  # 8
]


def get_exif_orientation(image: Image) -> int:
    """Returns exif orientation (1-8) or 0 if no exif data"""
    if hasattr(image, "_getexif"):
        exif = image._getexif()
        if exif is not None and EXIF_ORIENTATION in exif.keys():
            return exif[EXIF_ORIENTATION]
    return 0


def exif_rotate(image: Image) -> Image:
    """Rotates image based on exif orientation"""
    method = ROTATE[get_exif_orientation(image)]
    if method is not None:
        return image.transpose(method)
    return image


def crop(image: Image, width: int, height: int) -> Image:
    """Resize and then crop image to specified height and width"""
    # resize
    ratio = max(width / image.size[0], height / image.size[1])
    resized = image.resize(
        (int(round(image.size[0] * ratio)), int(round(image.size[1] * ratio))),
        Image.ANTIALIAS,
    )
    # crop
    half_width = int(round(width / 2))
    half_height = int(round(height / 2))
    center_width = int(round(resized.size[0] / 2))
    center_height = int(round(resized.size[1] / 2))
    return resized.crop(
        (
            center_width - half_width,
            center_height - half_height,
            center_width + half_width,
            center_height + half_height,
        )
    )


def read_image(filename: str) -> Image:
    return Image.open(filename)


def write_image(filename: str, image: Image) -> str:
    # remove alpha channel since not relevant to JPEG
    image = image.convert("RGB")
    image.save(filename, "JPEG")
    return filename


def get_cropped_ids() -> Set[int]:
    return set(
        [
            int(f.split(".")[0])
            for f in listdir(CROP_PATH)
            if path.isfile(path.join(CROP_PATH, f))
        ]
    )


def crop_recipes():
    i = 0
    if not path.exists(CROP_PATH):
        mkdir(CROP_PATH)
    cropped_recipes = get_cropped_ids()
    for f in listdir(IMAGE_PATH):
        recipe_id, ending = f.split(".")
        current_recipe_id = int(recipe_id)
        filename = path.join(IMAGE_PATH, f)
        if (
            current_recipe_id not in cropped_recipes
            and ending in IMAGE_ENDINGS
            and path.isfile(filename)
        ):
            try:
                photo = crop(
                    height=IMAGE_RESOLUTION,
                    width=IMAGE_RESOLUTION,
                    image=exif_rotate(Image.open(filename)),
                )
                write_image(
                    path.join(CROP_PATH, f"{current_recipe_id}.jpeg"),
                    photo,
                )
                i += 1
            except:
                continue
            if i % 500 == 0:
                print(f"cropped: ({i}) images")


if __name__ == "__main__":
    crop_recipes()
    image_recipe_ids = get_cropped_ids()
    with open("./data/02_filter.json", mode="r", encoding="utf-8") as i, open(
        "./data/04_crop.json", mode="w", encoding="utf-8"
    ) as o:
        for line in i:
            row = json.loads(line)
            if int(row["id"]) in image_recipe_ids:
                o.write(line)
