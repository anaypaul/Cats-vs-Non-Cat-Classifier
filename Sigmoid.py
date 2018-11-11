import numpy as np

def sigmoid(set_x):
    x = 1 / (1 + np.exp(-set_x))
    return x