from .Initializer import Initializer, np

class Zero(Initializer):
    def call(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)