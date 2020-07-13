from light.functions import multiply, mean, sum, negative
from ..activations import log_softmax

def softmax_crossentropy(y, label):
    l = log_softmax(y)
    m = multiply(l, label)
    n = mean(m, axis=0)
    p = sum(n)
    return negative(p)
