from light.nn.initializations import Uniform, Zero
from light.functions import variable, matmul, add
from .numpy import numpy as np

def linear(x, input_features, output_features, dtype=np.float32, init=Uniform()):
    w_shape = (input_features, output_features)
    b_shape = (output_features, )
    w = variable(init(w_shape, dtype=dtype))
    b = variable(Zero()(b_shape, dtype=dtype))
    xw = matmul(x, w)
    xwb = add(xw, b)
    return w, b, xwb
