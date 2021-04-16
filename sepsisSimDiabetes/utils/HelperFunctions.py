import numpy as np


def noisy(var, max_val, min_val=0):
    vals = np.array([max(min_val, var-1), var, min(max_val, var+1)])
    return np.random.choice(vals, p=np.array([0.05, 0.90, 0.05]))