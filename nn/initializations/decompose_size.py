from .Initializer import np

def decompose_size(size):
    fan_in, fan_out = None, None
    if len(size) == 2:
        fan_in, fan_out = size[0], size[1]
    elif len(size) == 4 or len(size) == 5:
        respectve_field_size = np.prod(size[2:])
        fan_in = np.multiply(size[1], respectve_field_size)
        fan_out = np.multiply(size[0], respectve_field_size)
    else:
        fan_in = fan_out = int(np.sqrt(np.prod(size)))
    
    return fan_in, fan_out