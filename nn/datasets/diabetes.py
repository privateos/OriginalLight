import os
import numpy as np

"""
from sklearn.datasets import load_diabetes
d = load_diabetes()
print(d)
m = {'x':d['data'], 'y':d['target']}
np.savez(os.path.join(os.path.dirname(__file__), 'diabetes.npz'), **m)

"""
def load_diabetes():
    filename = os.path.join(os.path.dirname(__file__), 'diabetes.npz')
    data = np.load(filename)
    return data['x'], data['y']
    
