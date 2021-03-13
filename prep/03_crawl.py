#!/usr/bin/env python

from enum import Enum
from glob import iglob
import json
from json.decoder import JSONDecodeError
import logging
from lxml.html import parse, HTMLParser
from os import mkdir, path
from random import randint, uniform
import requests
from shutil import copyfileobj
import sys
from time import sleep, time
from typing import Optional

CACHE_COUNTER = 0
JSON_LD = "//script[@type='application/ld+json']"
MIN_DELAY = 3.0
MAX_DELAY = 7.0
START_TIME = int(time())


class Encoding(Enum):
    FAILED = 0
    BMP = 1
    GIF = 2
    HTML = 3
    JPEG = 4
    PDF = 5
    PNG = 6
    SVG = 7
    WEBP = 8


class Kind(Enum):
    FAILED = 0
    IMAGE = 1
    RECIPE = 2


CONTENT_TYPE_TO_ENCODING = {
    "application/pdf": Encoding.PDF,
    "image/bmp": Encoding.BMP,
    "image/gif": Encoding.GIF,
    "image/jpeg": Encoding.JPEG,
    "image/png": Encoding.PNG,
    "image/svg+xml": Encoding.SVG,
    "image/webp": Encoding.WEBP,
    "text/html": Encoding.HTML,
}

ENCODING_TO_KIND = {
    Encoding.BMP: Kind.IMAGE,
    Encoding.GIF: Kind.IMAGE,
    Encoding.HTML: Kind.RECIPE,
    Encoding.JPEG: Kind.IMAGE,
    Encoding.PDF: Kind.IMAGE,
    Encoding.PNG: Kind.IMAGE,
    Encoding.SVG: Kind.IMAGE,
    Encoding.WEBP: Kind.IMAGE,
}


def log(message: str) -> str:
    hours = "{:.4f}".format((int(time()) - START_TIME) * 1.0 / (60 * 60))
    return f"{hours}h: {message}"


def save(recipe_id: int, kind: Kind, url: str) -> bool:
    prefix = f"./data/crawl/{kind.name.lower()}/{recipe_id}."
    for filepath in iglob(prefix + "*"):
        global CACHE_COUNTER
        CACHE_COUNTER += 1
        if CACHE_COUNTER % 100 == 0:
            logging.info(log(f"{CACHE_COUNTER} cached files: current file={filepath}"))
        return True

    logging.info(log(f"save: recipe_id={recipe_id} kind={kind} url={url}"))
    sleep(uniform(MIN_DELAY, MAX_DELAY))
    response = None
    try:
        response = requests.get(
            url,
            stream=True,
        )
    except:
        logging.error(log(f"unexpected error: url={url} error={sys.exc_info()}"))
        return False
    if response is None or response.status_code < 200 or response.status_code >= 300:
        logging.error(log(f"request failed: url={url} status={response.status_code}"))
        return False

    content_type = response.headers["Content-Type"]
    encoding = CONTENT_TYPE_TO_ENCODING[content_type.split(";", 1)[0].strip()]
    derived_kind = ENCODING_TO_KIND[encoding]
    if kind is not derived_kind:
        logging.error(
            log(
                f"expected kind={kind}, got kind={derived_kind} url={url} content_type={content_type} encoding={encoding}"
            )
        )
        return False

    # use x mode to ensure we don't overwrite files
    with open(prefix + encoding.name.lower(), mode="xb") as f:
        response.raw.decode_content = True
        # save response to local filesystem
        copyfileobj(response.raw, f)
    return True


def read_image(recipe_id: int) -> Optional[str]:
    tree = parse(
        f"./data/crawl/{Kind.RECIPE.name.lower()}/{recipe_id}.html",
        parser=HTMLParser(encoding="utf-8"),
    )
    elements = tree.xpath(JSON_LD)
    if len(elements) == 0:
        logging.debug(log(f"{filename}: no recipe"))
        return None
    for element in elements:
        j = None
        try:
            j = json.loads(element.text)
        except JSONDecodeError as e:
            logging.debug(log(f"{filename} invalid json: {element.text}"))
            return None
        if "@type" in j.keys():
            if j["@type"] == "Recipe":
                return j["image"].strip()
        if "@graph" in j.keys():
            for node in j["@graph"]:
                if node["@type"] == "Recipe":
                    return j["image"].strip()
        else:
            logging.debug(log(f"{filename}: no @graph or @type=Recipe element in json"))
            logging.debug(log(json.dumps(j, indent=2, ensure_ascii=False)))
            return None
    return None


if __name__ == "__main__":
    logging.basicConfig(
        filename=f"data/03_crawl.log",
        format="%(levelname)s:%(message)s",
        level=logging.INFO,
    )
    logging.info(log(f"start crawl"))
    current_recipe = 0
    current_image = 0
    total_recipes = 113955
    if not path.exists("./data/crawl"):
        mkdir("./data/crawl")
    for kind in Kind:
        kind_dir = f"./data/crawl/{kind.name.lower()}"
        if not path.exists(kind_dir):
            mkdir(kind_dir)
    with open("./data/02_filter.json", mode="r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            recipe_id = int(row["id"])
            filename = f"{recipe_id}.html"
            url = row["url"]
            if save(recipe_id=recipe_id, kind=Kind.RECIPE, url=f"https://{url}"):
                current_recipe += 1
                image_url = read_image(recipe_id=recipe_id)
                if image_url is not None:
                    if save(recipe_id=recipe_id, kind=Kind.IMAGE, url=image_url):
                        current_image += 1
            if current_recipe % 10 == 0:
                logging.info(
                    log(
                        f"saved {current_image} images, {current_recipe} recipes, {total_recipes} total"
                    )
                )
    logging.info(log(f"end crawl"))