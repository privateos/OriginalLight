from light.functions import subtract, abs, mean, sum

def mean_absoulte_error(y, label):
    m = subtract(y, label)
    a = abs(m)
    z = mean(a, axis=0)
    return sum(z)
