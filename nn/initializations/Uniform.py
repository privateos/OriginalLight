from .Initializer import Initializer, np

class Uniform(Initializer):
    def __init__(self, scale=0.05):
        self.scale = scale
    
    def call(self, shape, dtype=None):
        if dtype is not None:
            return np.random.uniform(-self.scale, self.scale, shape).astype(dtype)
        else:
            return np.random.uniform(-self.scale, self.scale, shape)
