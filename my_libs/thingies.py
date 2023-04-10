import numpy as np


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
