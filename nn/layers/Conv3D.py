from light.nn.initializations import Uniform, Zero
from light.functions import variable, zero_pad
from light.functions import conv3d as OriginalConv3d
from .numpy import numpy as np

"""
x.shape = (batch_size, D, H, W, C)
filter.shape = (FD, FH, FW, C, output_channel)
"""
def conv3d(x, D, H, W, C, FD, FH, FW, output_channel, strides, padding="VALID", dtype=np.float32, init=Uniform()):
    f_shape = (FD, FH, FW, C, output_channel)
    f = variable(init(f_shape, dtype=dtype))

    if padding == 'SAME':
        out_d = D//strides[0]
        if D%strides[0] != 0:
            out_d += 1
        out_h = H//strides[1]
        if H%strides[1] != 0:
            out_h += 1
        out_w = W//strides[2]
        if W%strides[2] != 0:
            out_w += 1
        
        pad_d = max((out_d - 1)*strides[0] + FD - D, 0)
        pad_h = max((out_h - 1)*strides[1] + FH - H, 0)
        pad_w = max((out_w - 1)*strides[2] + FW - W, 0)

        pad_d_top = pad_d//2
        pad_d_bottom = pad_d - pad_d_top

        pad_top = pad_h//2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w//2
        pad_right = pad_w - pad_left
        x = zero_pad(x, ((0, 0),(pad_d_top, pad_d_bottom), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
        return f, OriginalConv3d(x, f, stride_d=strides[0], stride_h=strides[1], stride_w=strides[2])
    elif padding == 'VALID':
        return f, OriginalConv3d(x, f, stride_d=strides[0], stride_h=strides[1], stride_w=strides[2])
    else:
        raise ValueError('padding must be VALID or SAME')
