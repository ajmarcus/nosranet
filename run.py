#!/usr/bin/env python

from nosranet.config import Label, Model
from nosranet.evaluate import evaluate, evaluate_baseline


if __name__ == "__main__":
    for l in Label:
        for m in Model:
            evaluate_baseline(label=l, name=m)
            evaluate(label=l, name=m, num_layers=3)