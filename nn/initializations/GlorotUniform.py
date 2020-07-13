from .Initializer import Initializer, np
from .decompose_size import decompose_size
from .Uniform import Uniform

class GlorotUniform(Initializer):
    def call(self, shape, dtype=None):
        fan_in, fan_out = decompose_size(shape)
        return Uniform(np.sqrt(6.0/(fan_in + fan_out)))(shpae, dtype)