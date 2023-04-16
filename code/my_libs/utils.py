from typing import Generator

import numpy as np
import time


def accentuate_weights(weights: list) -> list:
    # create a list to store the accentuated weights
    accentuated_weights = []
    # for each weight
    for weight in weights:
        # accentuate the weight
        accentuated_weight = weight ** 2
        # add the accentuated weight to the list
        accentuated_weights.append(accentuated_weight)
    # normalize the accentuated weights
    accentuated_weights = accentuated_weights / np.sum(accentuated_weights)
    # return the accentuated weights
    return accentuated_weights


def clock() -> Generator[float, None, None]:
    # first tick
    tick = time.time()
    yield 0.0

    while True:
        # calculate the time since the last tick
        delta = time.time() - tick
        # save the current tick
        tick = time.time()
        # yield the time since the last tick
        yield delta
