import numpy as np
from numba import jit


@jit
def calc_mse(src: np.ndarray, res: np.ndarray):
    return ((res - src) ** 2).mean()
