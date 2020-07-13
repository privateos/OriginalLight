from .Initializer import Initializer, np

class One(Initializer):
    def call(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype)