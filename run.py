#!/usr/bin/env python

from nosranet.config import Label, Model
from nosranet.evaluate import evaluate, evaluate_baseline


if __name__ == "__main__":
    num_layers = [3, 15, 45]
    dropout_prob = [0.03, 0.3, 0.6, 0.9]
    for l in Label:
        for m in Model:
            evaluate_baseline(label=l, name=m)
            for nl in num_layers:
                for d in dropout_prob:
                    evaluate(label=l, name=m, num_layers=nl, dropout_prob=d, epochs=75)