from light.backend import backend as np
#pickle, paradox

class Module(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError
    
    def update(self, *args, **kwargs):
        raise NotImplementedError
    
    def predict(self, input):
        raise NotImplementedError

    def save(self, param_dict, filename):
        np.savez(filename, **param_dict)

    def load(self, filename):
        import os
        if os.path.exists(filename):
            return np.load(filename)
        else:
            return None

    def category(self, onehot):
        pl = np.argmax(onehot, axis=1)
        return pl

    #pred, labels 均为onehot
    def accuracy(self, pred, labels):
        y1 = np.argmax(pred, axis=1)
        y2 = np.argmax(labels, axis=1)
        a = y1==y2
        return np.mean(a)

    def mean_squared_error(self, pred, labels):
        pl = np.subtract(pred, labels)
        pl = np.square(pl)
        pl = np.mean(pl, axis=0)
        return np.sum(pl)

    def mean_absolute_error(self, pred, labels):
        pl = np.subtract(pred, labels)
        pl = np.abs(pl)
        pl = np.mean(pl, axis=0)
        return np.sum(pl)
