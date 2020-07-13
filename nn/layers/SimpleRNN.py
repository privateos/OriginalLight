from light.nn.initializations import Uniform, Zero
from light.functions import variable, matmul, add, getitem, tanh, expand_dims, concatenate
from .numpy import numpy as np

#x.shape = (batch_size, time_step, input_features)
#(batch_size, time_step, output_features)
"""
h_t = tanh(h_t_1@U + X_t@W + b)
"""
def simple_rnn(x, time_step, input_features, output_features, dtype=np.float32, init=Uniform()):
    output = []
    bias = Zero()
    U = variable(init((output_features, output_features), dtype=dtype))
    W = variable(init((input_features, output_features), dtype=dtype))
    b = variable(bias((output_features, ), dtype=dtype))
    X_t = getitem(x, (slice(None, None, None), 0, slice(None, None, None)))
    t = tanh(add(matmul(X_t, W), b))
    output.append(expand_dims(t, axis=1))

    h_t_1 = t
    for i in range(1, time_step):
        X_t = getitem(x, (slice(None, None, None), i, slice(None, None, None)))
        t = tanh(add(matmul(h_t_1, U), add(matmul(X_t, W), b)))
        output.append(expand_dims(t, axis=1))
        h_t_1 = t

    y = concatenate(output, axis=1)#(batch_size, time_step, output_features)
    return U, W, b, y

    

    