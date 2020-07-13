from .Initializer import Initializer, np
from .decompose_size import decompose_size
from .Normal import Normal

class HeNormal(Initializer):
    def call(self, shape, dtype=None):
        fan_in, fan_out = decompose_size(shape)
        return Normal(np.sqrt(2.0/fan_in))(shape)