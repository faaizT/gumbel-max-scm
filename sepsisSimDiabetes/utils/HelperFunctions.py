import numpy as np


def noisy(var, max_val, min_val=0):
    probs = np.ones(max_val + 1 - min_val) * 0.05
    probs[var] = 1 - 0.05 * max_val
    return np.random.choice(range(min_val, max_val + 1), p=probs)
