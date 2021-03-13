#!/usr/bin/env python

# CSV with labels from 01_join.json

from collections import Counter
import json
from numpy import median
from typing import Counter

MIN_REVIEWS = 3
MAX_INGREDIENTS = 4096
AVG_REVIEWS = set([0, 1, 2, 3, 5])  # keep all reviews except 4
count_review: Counter = Counter()
text_review = []
max_reviews = 0
num_negative = 0
num_positive = 0
num_recipes = 0
num_reviews = 0
sample_positive = 0
output_negative = 0
output_positive = 0

if __name__ == "__main__":
    with open("./data/01_join.json", mode="r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            reviews = int(row["num_reviews"])
            avg_review = int(row["avg_round_review"])
            if (
                len(row["title"]) > 2
                and len(row["ingredients"]) > 2
                and len(row["steps"]) > 2
                and reviews >= MIN_REVIEWS
                and avg_review in AVG_REVIEWS
            ):
                text = []
                text.append(row["title"])
                text.extend(row["ingredients"])
                text.extend(row["steps"])
                text_review.append((" ".join(text), avg_review))
                count_review.update([avg_review])
                max_reviews = max(max_reviews, reviews)
                num_recipes += 1
                num_reviews += reviews

    for review, count in count_review.items():
        if review == 5:
            num_positive += count
        else:
            num_negative += count

    with open("./data/02_full.csv", mode="w", encoding="utf-8") as f:
        for index, text_review in enumerate(text_review):
            text, avg_review = text_review
            # label
            if avg_review == 5:
                label = 1
                sample_positive += 1
            else:
                label = 0
            row = f'"{text}",{label}\n'
            if index % 10000 == 0:
                print(f"{row}")
            if label == 0 or sample_positive <= num_negative * 2:
                f.write(row)
                if label == 0:
                    output_negative += 1
                else:
                    output_positive += 1
    print(f"count_review: {count_review}")
    print(f"num_recipes: {num_recipes}")
    print(f"num_reviews: {num_reviews}")
    print(f"max_reviews: {max_reviews}")
    print(f"min_reviews: {MIN_REVIEWS}")
    print(f"num_negative: {num_negative}")
    print(f"num_positive: {num_positive}")
    print(f"sample_positive: {sample_positive}")
    print(f"output_negative: {output_negative}")
    print(f"output_positive: {output_positive}")