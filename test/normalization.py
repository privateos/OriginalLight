import sys
sys.path.append('D:/repo')

import light as F
from light import backend as np
import light.functions as graph

cond = graph.constant(False)
x = graph.placeholder()
mean = graph.variable(np.random.randn(5))
std = graph.variable(np.random.randn(5))
momentum = graph.constant(0.9)
eps = graph.constant(np.finfo(np.float64).eps)
x_hat = graph.normalization(cond, x, mean, std, momentum, eps)
y = graph.mean(x_hat)

forward = graph.executor(*[y])
backward = graph.gradient(y, *[x, x_hat])


test_x_hat = graph.divide(graph.subtract(x, mean), std)
test_y = graph.mean(test_x_hat)
test_forward = graph.executor(*[test_y])
test_backward = graph.gradient(test_y, *[x, test_x_hat])


x_shape = (10, 5)
for i in range(10):
    x_v = np.random.uniform(-1.0, 1.0, x_shape)
    feed_dict = {x:x_v}
    test_y_v, = test_forward.run(feed_dict=feed_dict)
    test_grad_x, test_grad_x_hat = test_backward.get()
    y_v, = forward.run(feed_dict=feed_dict)
    grad_x, grad_x_hat = backward.get()


    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(np.max(np.abs(y_v - test_y_v)))
    print(np.max(np.abs(test_grad_x - grad_x)))
    print(np.max(np.abs(test_grad_x_hat - grad_x_hat)))
    input()
