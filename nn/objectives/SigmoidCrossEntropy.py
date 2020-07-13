from ..activations import sigmoid
from .BinaryCrossEntropy import binary_crossentropy

def sigmoid_crossentropy(y, label):
    y = sigmoid(y)
    return binary_crossentropy(y, label)
