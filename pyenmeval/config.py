import numpy as np
import random

DEFAULT_RANDOM_STATE = 42

def set_seed(seed=DEFAULT_RANDOM_STATE):
    np.random.seed(seed)
    random.seed(seed)