import os
import numpy as np

"""
import scipy.io

minst_data = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'usps_train.mat'))
x = minst_data['usps_train']
usps = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'usps_train_labels.mat'))
y = usps['usps_train_labels']
m = {'x':x, 'y':y}
np.savez(os.path.join(os.path.dirname(__file__), 'usps.npz'), **m)

"""
def load_usps():
    filename = os.path.join(os.path.dirname(__file__), 'usps.npz')
    data = np.load(filename)
    return data['x'], data['y']
