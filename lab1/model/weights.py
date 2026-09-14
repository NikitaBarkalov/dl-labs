import numpy as np

def generate_weights(rows, cols, init_type, rng):
    if init_type == 'He':
        return rng.normal(0, np.sqrt(2 / rows), (rows, cols))

    if init_type == 'Xavier':
        return rng.normal(0, np.sqrt(2 / (rows + cols)), (rows, cols))

    raise Exception('Invalid init_type')