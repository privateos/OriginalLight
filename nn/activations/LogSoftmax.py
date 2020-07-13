from light.functions import exp, sum, divide, max, subtract, log

def log_softmax(x):
    m = max(x, axis=-1, keepdims=True)
    y = subtract(x, m)
    e = exp(y)
    s = sum(e, axis=-1, keepdims=True)
    l = log(s)
    return subtract(y, l)
