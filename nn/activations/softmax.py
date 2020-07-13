from light.functions import exp, sum, divide

def softmax(x):
    e = exp(x)
    t = sum(e, axis=-1, keepdims=True)
    return divide(e, t)

