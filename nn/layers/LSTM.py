from light.nn.initializations import Uniform, Zero
from light.functions import variable, matmul, add, getitem, tanh, sigmoid, expand_dims, concatenate, multiply
from .numpy import numpy as np

#x.shape = (batch_size, time_step, input_features)
#output.shape = (batch_size, time_step, output_features)
"""
C_t_hat = tanh(X_t@W_c + h_t_1@U_c + b_c)
i_t = sigmoid(X_t@W_i + h_t_1@U_i + b_i)
f_t = sigmoid(X_t@W_f + h_t_1@U_f + b_f)
o_t = sigmoid(X_t@W_o + h_t_1@U_o + b_o)
C_t = f_t*C_t_1 + i_t*C_t_hat
"""
def lstm(x, time_step, input_features, output_features, dtype=np.float32, init=Uniform()):
    output = []
    bias = Zero()
    w_shape = (input_features, output_features)
    u_shape = (output_features, output_features)
    b_shape = (output_features, )
    ######C_t_hat##############
    W_c = variable(init(w_shape, dtype=dtype))
    U_c = variable(init(u_shape, dtype=dtype))
    b_c = variable(bias(b_shape, dtype=dtype))

    ######i_t#################
    W_i = variable(init(w_shape, dtype=dtype))
    U_i = variable(init(u_shape, dtype=dtype))
    b_i = variable(bias(b_shape, dtype=dtype))

    #####f_t##################
    W_f = variable(init(w_shape, dtype=dtype))
    U_f = variable(init(u_shape, dtype=dtype))
    b_f = variable(bias(b_shape, dtype=dtype))

    ####o_t##################
    W_o = variable(init(w_shape, dtype=dtype))
    U_o = variable(init(u_shape, dtype=dtype))
    b_o = variable(bias(b_shape, dtype=dtype))

    X_t = getitem(x, (slice(None, None, None), 0, slice(None, None, None)))
    C_t_hat = tanh(add(matmul(X_t, W_c), b_c))
    i_t = sigmoid(add(matmul(X_t, W_i), b_i))
    #f_t = sigmoid(add(matmul(X_t, W_f), b_f))
    o_t = sigmoid(add(matmul(X_t, W_o), b_o))
    C_t = multiply(i_t, C_t_hat)
    h_t = multiply(o_t, tanh(C_t))
    output.append(expand_dims(h_t, axis=1))

    C_t_1 = C_t
    h_t_1 = h_t
    for i in range(1, time_step):
        X_t = getitem(x, (slice(None, None, None), i, slice(None, None, None)))
        C_t_hat = tanh(add(matmul(X_t, W_c), add(matmul(h_t_1, U_c), b_c)))
        i_t = sigmoid(add(matmul(X_t, W_i), add(matmul(h_t_1, U_i), b_i)))
        f_t = sigmoid(add(matmul(X_t, W_f), add(matmul(h_t_1, U_f), b_f)))
        o_t = sigmoid(add(matmul(X_t, W_o), add(matmul(h_t_1, U_o), b_o)))
        C_t = add(multiply(f_t, C_t_1), multiply(i_t, C_t_hat))
        h_t = multiply(o_t, tanh(C_t))

        C_t_1 = C_t
        h_t_1 = h_t
        output.append(expand_dims(h_t, axis=1))
    
    y = concatenate(output, axis=1)
    return W_c, U_c, b_c, W_i, U_i, b_i, W_f, U_f, b_f, W_o, U_o, b_o, y