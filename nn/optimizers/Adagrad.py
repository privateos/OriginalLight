from .OptimizerBase import OptimizerBase, np

class Adagrad(OptimizerBase):
    """Adagrad updates

    Scale learning rates by dividing with the square root of accumulated
    squared gradients. See [1]_ for further description.

    Parameters
    ----------
    epsilon : float
        Small value added for numerical stability.

    Notes
    -----
    Using step size eta Adagrad calculates the learning rate for feature i at
    time step t as:

    .. math:: \\eta_{t,i} = \\frac{\\eta}
       {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}

    as such the learning rate is monotonically decreasing.

    Epsilon is not included in the typical formula, see [2]_.

    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.

    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    """

    def __init__(self, scheduler, epsilon=1e-6):
        super(self.__class__, self).__init__(scheduler)

        self.epsilon = epsilon

        self.cache = {}

    def update(self, fn, param, param_grad, cur_loss=None):
        lr = self.scheduler(self.cur_step, cur_loss)
        cache = self.cache
        eps = self.epsilon

        for p, g in zip(param, param_grad):
            c = cache.get(p)
            if c is None:
                c = np.square(g)
            else:
                c = np.add(c, np.square(g))
            fn(p, np.divide(np.multiply(lr, g), np.add(np.sqrt(c), eps)))
            cache[p] = c

        """
        # init cache
        if self.cache is None:
            self.cache = [_zero(g.shape) for g in grads]

        # update parameters
        for i, (c, p, g) in enumerate(zip(self.cache, params, grads)):
            c += np.power(g, 2)
            p -= self.lr * g / (np.sqrt(c) + self.epsilon)
            self.cache[i] = c

        super(Adagrad, self).update(params, grads)
        """