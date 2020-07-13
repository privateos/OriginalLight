import os
import numpy as np

"""
from sklearn.datasets import load_boston
d = load_boston()
m = {'x':d['data'], 'y':d['target']}
np.savez(os.path.join(os.path.dirname(__file__), 'boston.npz'), **m)

"""
def load_boston():
    filename = os.path.join(os.path.dirname(__file__), 'boston.npz')
    data = np.load(filename)
    return data['x'], data['y']
    
