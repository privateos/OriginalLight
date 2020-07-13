from .Initializer import Initializer, np

class Orthogonal(Initializer):
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2.0)
        self.gain = gain
    
    def call(self, shape, dtype=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.nprandom.normal(loc=0.0, scale=1.0, size=flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        q = np.multiply(self.gain, q)
        if dtype is not None:
            return q.astype(dtype)
        else:
            return q