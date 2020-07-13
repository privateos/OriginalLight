import os
import numpy as np

#from sklearn.datasets import load_iris
#d = load_iris()
#m = {'x':d['data'], 'y':d['target']}
#np.savez(os.path.join(os.path.dirname(__file__), 'iris.npz'), **m)

def load_iris():
    filename = os.path.join(os.path.dirname(__file__), 'iris.npz')
    data = np.load(filename)
    return data['x'], data['y']
