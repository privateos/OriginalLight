import os
import numpy as np

"""
from sklearn.datasets import load_digits
d = load_digits()
print(d)
m = {'x':d['data'], 'y':d['target']}
np.savez(os.path.join(os.path.dirname(__file__), 'digits.npz'), **m)

"""
def load_digits():
    filename = os.path.join(os.path.dirname(__file__), 'digits.npz')
    data = np.load(filename)
    return data['x'], data['y']
    
