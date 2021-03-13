#!/usr/bin/env python

# Join ingredients and reviews datasets
# Ingredients from "RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation"
# https://www.aclweb.org/anthology/2020.inlg-1.4/
# Reviews from "Generating Personalized Recipes from Historical User Preferences"
# https://www.aclweb.org/anthology/D19-1613/

import csv
import json
from json.decoder import JSONDecodeError
from os import mkdir, path
from typing import Dict

HAS_INPUT = path.exists("./data/reviews.csv") and path.exists("./data/ingredients.csv")
DATA_DIR = "./data/"
INPUT_DIR = DATA_DIR if HAS_INPUT else "./example/"
reviews: Dict[int, list] = {}


def d(row, index):
    try:
        return json.loads(row[index])
    except JSONDecodeError as e:
        print(f"json:({row[index]})")
        print(f"index:({index}),row:({row})")
        raise e


if __name__ == "__main__":
    if not path.exists(DATA_DIR):
        mkdir(DATA_DIR)
    with open(INPUT_DIR + "reviews.csv", mode="r", encoding="utf-8") as f:
        recipe = csv.reader(f)
        first = True
        for row in recipe:
            if first == True:
                first = False
                continue
            recipe_id = int(row[1])
            review = int(row[3])
            if recipe_id in reviews.keys():
                reviews[recipe_id].append(review)
            else:
                reviews[recipe_id] = [review]

    with open(INPUT_DIR + "ingredients.csv", mode="r", encoding="utf-8") as f, open(
        DATA_DIR + "01_join.json", mode="w", encoding="utf-8"
    ) as o:
        recipe = csv.reader(f)
        for index, row in enumerate(recipe):
            if index % 10000 == 0:
                print(f"row: {index}")
            url = row[4]
            if row[5] == "Gathered" and url.startswith("www.food.com"):
                recipe_id = int(url.split("-")[-1])
                if recipe_id in reviews.keys():
                    score = sum(reviews[recipe_id])
                    if score > 0:
                        entities = set([e.lower() for e in d(row, 6)])
                        out = {
                            "id": recipe_id,
                            "title": row[1],
                            "ingredients": d(row, 2),
                            "steps": d(row, 3),
                            "url": url,
                            "entities": sorted(list(entities)),
                            "reviews": reviews[recipe_id],
                            "avg_review": score * 1.0 / len(reviews[recipe_id]),
                            "avg_round_review": round(
                                score * 1.0 / len(reviews[recipe_id])
                            ),
                            "num_entities": len(entities),
                            "num_reviews": len(reviews[recipe_id]),
                            "score": score,
                        }
                        if index % 10000 == 0:
                            print(f"{out}")
                        o.write(json.dumps(out, ensure_ascii=False) + "\n")
