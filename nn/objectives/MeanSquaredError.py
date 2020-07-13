from light.functions import subtract, square, mean, sum

def mean_squared_error(y, label):
    m = subtract(y, label)
    s = square(m)
    z = mean(s, axis=0)
    return sum(z)
