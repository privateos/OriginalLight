from light.backend import backend as np

class Initializer(object):
    def __call__(self, shape, dtype=None):
        return self.call(shape, dtype)
    
    def call(self, shape, dtype=None):
        raise NotImplementedError