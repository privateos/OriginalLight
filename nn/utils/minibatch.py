from .numpy import numpy as np

def minibatch(X, batch_size=256, shuffle=True):
    N = X.shape[0]
    ix = np.arange(N)
    n_batches = int(np.ceil(N/batch_size))
    if shuffle:
        np.random.shuffle(ix)
    
    def generator():
        for i in range(n_batches):
            yield ix[i*batch_size:(i+1)*batch_size]
    
    return generator(), n_batches

