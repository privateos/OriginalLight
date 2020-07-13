import os
import numpy as np
"""
def load_MNIST(file):
    file = np.load(file)
    X = file['X']
    Y = file['Y']
    return X, Y
"""

def load_mnist():
    d = np.load(os.path.join(os.path.dirname(__file__), 'mnist.npz'))
    return d['x'], d['y']


