from light.nn.initializations import Uniform, Zero
from light.functions import variable, zero_pad
from light.functions import conv1d as OriginalConv1d
from .numpy import numpy as np

"""
x.shape = (batch_size, W, C)
filter.shape = (FW, C, output_channel)
"""
def conv1d(x, W, C, FW, output_channel, stride, padding="VALID", dtype=np.float32, init=Uniform()):
    f_shape = (FW, C, output_channel)
    f = variable(init(f_shape, dtype=dtype))

    if padding == 'SAME':
        out_w = W//stride
        if W%stride != 0:
            out_w += 1
        pad_w = max((out_w - 1)*stride + FW - W, 0)
        pad_left = pad_w//2
        pad_right = pad_w - pad_left
        x = zero_pad(x, ((0, 0), (pad_left, pad_right), (0, 0)))
        return f, OriginalConv1d(x, f, stride=stride)
    elif padding == 'VALID':
        return f, OriginalConv1d(x, f, stride=stride)
    else:
        raise ValueError('padding must be VALID or SAME')

    
