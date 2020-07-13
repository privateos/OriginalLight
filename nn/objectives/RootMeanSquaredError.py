from .MeanSquaredError import mean_squared_error
from light.functions import sqrt

def root_mean_squared_error(y, label):
    return sqrt(mean_squared_error(y, label))
