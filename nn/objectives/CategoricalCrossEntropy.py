from light.functions import log, multiply, mean, sum, negative

def categorical_crossentropy(y, label):
    l = log(y)
    m = multiply(label, l)
    n = mean(m, axis=0)
    p = sum(n)
    return negative(p)
