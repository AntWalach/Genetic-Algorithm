
import numpy as np

def initialize_real_population(pop_size, n_vars, low, high):
    return [np.random.uniform(low, high, size=n_vars) for _ in range(pop_size)]

def clip_to_bounds(vec, low, high):
    return np.clip(vec, low, high)