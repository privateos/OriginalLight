from light.nn.initializations import Uniform, Zero
from light.functions import variable, constant, matmul, add, subtract, multiply, getitem, tanh, sigmoid, expand_dims, concatenate
from .numpy import numpy as np

#x.shape = (batch_size, time_step, input_features)
#output.shape = (batch_size, time_step, output_features)
"""
z_t = sigmoid(X_t@W_z + h_t_1@U_z)
r_t = sigmoid(X_t@W_r + h_t_1@U_r)
h_t_hat = tanh(X_t@W_t + (r_t*h_t_1)@U_t)
h_t = (1-z_t)*h_t_1 + z_t*h_t_hat
"""
def gru(x, time_step, input_features, output_features, dtype=np.float32, init=Uniform()):
    output = []

    w_shape = (input_features, output_features)
    u_shape = (output_features, output_features)

    W_z = variable(init(w_shape, dtype=dtype))
    U_z = variable(init(u_shape, dtype=dtype))
    W_r = variable(init(w_shape, dtype=dtype))
    U_r = variable(init(u_shape, dtype=dtype))
    W_t = variable(init(w_shape, dtype=dtype))
    U_t = variable(init(u_shape, dtype=dtype))
    const_1 = constant(np.array(1.0, dtype=dtype))

    X_t = getitem(x, (slice(None, None, None), 0, slice(None, None, None)))
    z_t = sigmoid(matmul(X_t, W_z))
    #r_t = sigmoid(matmul(X_t, W_r))
    h_t_hat = tanh(matmul(X_t, W_t))
    h_t = multiply(z_t, h_t_hat)
    output.append(expand_dims(h_t, axis=1))
    h_t_1 = h_t
    for i in range(1, time_step):
        X_t = getitem(x, (slice(None, None, None), i, slice(None, None, None)))
        z_t = sigmoid(add(matmul(X_t, W_z), matmul(h_t_1, U_z)))
        r_t = sigmoid(add(matmul(X_t, W_r), matmul(h_t_1, U_r)))
        h_t_hat = tanh(add(matmul(X_t, W_t), matmul(multiply(r_t, h_t_1), U_t)))
        h_t = add(multiply(subtract(const_1, z_t), h_t_1), multiply(z_t, h_t_hat))

        output.append(expand_dims(h_t, axis=1))
        h_t_1 = h_t

    y = concatenate(output, axis=1)
    return W_z, U_z, W_r, U_r, W_t, U_t, y