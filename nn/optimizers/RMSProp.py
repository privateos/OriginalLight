from .OptimizerBase import OptimizerBase, np

class RMSProp(OptimizerBase):
    """RMSProp updates

    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.

    Parameters
    ----------
    rho : float
        Gradient moving average decay factor.
    epsilon : float
        Small value added for numerical stability.

    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.

    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:

    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """

    def __init__(self, scheduler, rho=0.9, epsilon=1e-6):
        super(self.__class__, self).__init__(scheduler)

        self.rho = rho
        self.epsilon = epsilon

        self.cache = {}
        self.iterations = 0

    def update(self, fn, param, param_grad, cur_loss=None):
        lr = self.scheduler(self.cur_step, cur_loss)
        cache = self.cache
        rho = self.rho
        eps = self.epsilon

        for p, g in zip(param, param_grad):
            c = cache.get(p)
            if c is None:
                c = np.multiply(1.0 - rho, np.square(g)) 
            else:
                c = np.add(np.multiply(rho, c), np.multiply(1.0 - rho, np.square(g)))
            fn(p, np.divide(np.multiply(lr, g), np.sqrt(np.add(c, eps))))  
            cache[p] = c
        """
        # init cache
        if self.cache is None:
            self.cache = [_zero(p.shape) for p in params]

        # update parameters
        for i, (c, p, g) in enumerate(zip(self.cache, params, grads)):
            c = self.rho * c + (1 - self.rho) * np.power(g, 2)
            p -= (self.lr * g / np.sqrt(c + self.epsilon))
            self.cache[i] = c
        """
