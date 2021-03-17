#!/usr/bin/env python

from nosranet.config import Label, Model
from nosranet.evaluate import evaluate
from sys import argv


if __name__ == "__main__":
    evaluate()
    # for l in Label:
    #     for m in Model:
    #         evaluate(label=l, name=m, num_layers=0)
    #         evaluate(label=l, name=m, num_layers=3)