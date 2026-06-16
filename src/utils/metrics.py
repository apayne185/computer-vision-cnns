import numpy as np


def maximum_precisions(precisions):
    """Interpolated precision envelope used for mAP calculation."""
    return np.flip(np.maximum.accumulate(np.flip(precisions)))


def mean_average_precision(precisions):
    return maximum_precisions(np.asarray(precisions)).mean()
