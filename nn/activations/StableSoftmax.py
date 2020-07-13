from light.functions import exp, sum, divide, max, subtract

def stable_softmax(x):
    y = max(x, axis=-1, keepdims=True)
    z = subtract(x, y)
    e = exp(z)
    t = sum(e, axis=-1, keepdims=True)
    return divide(e, t)