import numpy as np


def cell_array(values):
    inner = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        inner[0, index] = value

    outer = np.empty((1,), dtype=object)
    outer[0] = inner
    return outer
