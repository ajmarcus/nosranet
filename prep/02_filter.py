#!/usr/bin/env python

# High level statistics from 01_join.json

from collections import Counter
import json
from typing import Counter, FrozenSet, List, NamedTuple

MIN_REVIEWS = 2
MAX_INGREDIENTS = 8192
AVG_REVIEWS = set([0, 1, 2, 3, 5])  # keep all reviews except 4


class Recipe(NamedTuple):
    id: int
    title: str
    url: str
    ingredients: FrozenSet[str]
    num_reviews: int
    avg_review: int


all_ingredients: Counter = Counter()
count_review: Counter = Counter()
recipe_reviews: List[Recipe] = []
max_ingredients = 0
max_reviews = 0
min_ingredients = 1
num_negative = 0
num_positive = 0
recipes = 0
reviews = 0
ouput = []
out_lines = 0

if __name__ == "__main__":
    with open("./data/01_join.json", mode="r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ingredients = frozenset(row["entities"])
            ingredients_size = len(ingredients)
            num_reviews = len(row["reviews"])
            avg_review = int(row["avg_round_review"])
            if (
                ingredients_size > 0
                and num_reviews >= MIN_REVIEWS
                and avg_review in AVG_REVIEWS
            ):
                all_ingredients.update(ingredients)
                recipe_reviews.append(
                    Recipe(
                        id=int(row["id"]),
                        title=row["title"],
                        url=row["url"],
                        ingredients=ingredients,
                        num_reviews=num_reviews,
                        avg_review=avg_review,
                    )
                )
                max_reviews = max(max_reviews, reviews)

    top_ingredients_list = [
        name for name, count in all_ingredients.most_common(MAX_INGREDIENTS)
    ]
    with open("./data/02_tokens.json", mode="w", encoding="utf-8") as f:
        json.dump({"tokens": top_ingredients_list}, f, ensure_ascii=False)
    top_ingredients = frozenset(top_ingredients_list)

    for recipe in recipe_reviews:
        current_ingredients = recipe.ingredients & top_ingredients
        ingredients_size = len(current_ingredients)
        if ingredients_size > 0:
            count_review.update([recipe.avg_review])
            ouput.append((current_ingredients, recipe))

    with open("./data/02_filter.json", mode="w", encoding="utf-8") as f:
        for ingredients, recipe in ouput:
            # label
            if recipe.avg_review == 5:
                label = 1
                num_positive += 1
            else:
                label = 0
                num_negative += 1
            recipes += 1
            reviews += recipe.num_reviews
            max_ingredients = max(max_ingredients, ingredients_size)
            min_ingredients = min(min_ingredients, ingredients_size)
            row = {
                "id": recipe.id,
                "title": recipe.title,
                "url": recipe.url,
                "ingredients": list(ingredients),
                "label": label,
                "avg_review": recipe.avg_review,
            }
            if recipes % 10000 == 0:
                print(f"{row}")
            # if label == 0 or recipes % 2 == 0:
            if True:
                out_lines += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"count_review: {count_review}")
    print(f"out_lines: {out_lines}")
    print(f"recipes: {recipes}")
    print(f"reviews: {reviews}")
    print(f"max_ingredients: {max_ingredients}")
    print(f"min_ingredients: {min_ingredients}")
    print(f"max_reviews: {max_reviews}")
    print(f"min_reviews: {MIN_REVIEWS}")
    print(f"num_negative: {num_negative}")
    print(f"num_positive: {num_positive}")
    print(f"all_ingredients_size: {len(all_ingredients)}")
    print(f"most common ingredients: {all_ingredients.most_common(10)}")