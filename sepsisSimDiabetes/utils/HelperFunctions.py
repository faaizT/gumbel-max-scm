import numpy as np

# TODO: make this consistent with model
def noisy(var, max_val, min_val=0):
    return np.random.choice(
        [max(min_val, var - 1), var, min(max_val, var + 1)], p=[0.05, 0.9, 0.05]
    )