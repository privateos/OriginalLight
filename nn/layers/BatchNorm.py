from light.nn.initializations import One, Zero
from light.functions import variable, constant, branch, multiply, add, mean, subtract, square, sqrt
from .numpy import numpy as np

#如果是(N, input_features) 则 batch_shape = (input_features)
#如果是(N, H, W, C) 则 batch_shape = (H, W, C)
#总之batch_shape = shape[1:]
def batch_norm(x, train_or_test, batch_shape, momentum=0.99, dtype=np.float32, eps=np.finfo(np.float32).eps):
    pass
