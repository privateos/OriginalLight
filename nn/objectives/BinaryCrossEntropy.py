from light.functions import constant, subtract, add
from .CategoricalCrossEntropy import categorical_crossentropy
from .numpy import numpy as np

def binary_crossentropy(y, label):
    l0 = categorical_crossentropy(y, label)
    const = constant(np.array(1.0))
    new_label = subtract(const, label)
    new_y = subtract(const, y)
    l1 = categorical_crossentropy(new_y, new_label)
    return add(l0, l1)
