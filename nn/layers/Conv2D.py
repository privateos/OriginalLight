from light.nn.initializations import Uniform, Zero
from light.functions import variable, zero_pad
from light.functions import conv2d as OriginalConv2d
from .numpy import numpy as np

"""
x.shape = (batch_size, H, W, C)
filter.shape = (FH, FW, C, output_channel)
"""
def conv2d(x, H, W, C, FH, FW, output_channel, strides, padding="VALID", dtype=np.float32, init=Uniform()):
    f_shape = (FH, FW, C, output_channel)
    f = variable(init(f_shape, dtype=dtype))
    if padding == 'SAME':
        out_h = H//strides[0]
        if H%strides[0] != 0:
            out_h += 1
        out_w = W//strides[1]
        if W%strides[1] != 0:
            out_w += 1
        pad_h = max((out_h - 1)*strides[0] + FH - H, 0)
        pad_w = max((out_w - 1)*strides[1] + FW - W, 0)
        pad_top = pad_h//2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w//2
        pad_right = pad_w - pad_left
        x = zero_pad(x, ((0, 0),(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
        return f, OriginalConv2d(x, f, stride_h=strides[0], stride_w=strides[1])
    elif padding == 'VALID':
        return f, OriginalConv2d(x, f, stride_h=strides[0], stride_w=strides[1])
    else:
        raise ValueError('padding must be VALID or SAME')

    
