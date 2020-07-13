from .Initializer import Initializer, np
from .decompose_size import decompose_size
from .Uniform import Uniform

class LecunUniform(Initializer):
    def call(self, shape, dtype=None):
        fan_in, fan_out = decompose_size(shape)
        return Uniform(np.sqrt(3.0/fan_in))(shape, dtype)
