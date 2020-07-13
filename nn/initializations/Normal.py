from .Initializer import Initializer, np

class Normal(Initializer):
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def call(self, shape, dtype=None):
        if dtype is not None:
            return np.random.normal(loc=self.mean, scale=self.std, size=shape).astype(dtype)
        else:
            return np.random.normal(loc=self.mean, scale=self.std,size=shape)
