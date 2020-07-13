import os
import numpy as np

"""
from sklearn.datasets import load_linnerud
d = load_linnerud()
#print(d)
m = {'x':d['data'], 'y':d['target']}
np.savez(os.path.join(os.path.dirname(__file__), 'linnerud.npz'), **m)

"""
def load_linnerud():
    filename = os.path.join(os.path.dirname(__file__), 'linnerud.npz')
    data = np.load(filename)
    return data['x'], data['y']
    
