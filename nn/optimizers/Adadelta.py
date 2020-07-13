from .OptimizerBase import OptimizerBase, np

class Adadelta(OptimizerBase):
    """ Adadelta updates

    Scale learning rates by the ratio of accumulated gradients to accumulated
    updates, see [1]_ and notes for further description.

    Parameters
    ----------
    rho : float
        Gradient moving average decay factor.
    epsilon : float
        Small value added for numerical stability.
    decay : float
        Decay parameter for the moving average.

    Notes
    -----
    rho should be between 0 and 1. A value of rho close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.

    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
    work for multiple datasets (MNIST, speech).

    In the paper, no learning rate is considered (so learning_rate=1.0).
    Probably best to keep it at this value.
    epsilon is important for the very first update (so the numerator does
    not become 0).

    Using the step size eta and a decay factor rho the learning rate is
    calculated as:

    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
                             {\sqrt{r_t + \epsilon}}\\\\
       s_t &= \\rho s_{t-1} + (1-\\rho)*(\\eta_t*g)^2

    References
    ----------
    .. [1] Zeiler, M. D. (2012):
           ADADELTA: An Adaptive Learning Rate Method.
           arXiv Preprint arXiv:1212.5701.
    """

    def __init__(self, scheduler, rho=0.9, epsilon=1e-6):
        super(self.__class__, self).__init__(scheduler)

        self.rho = rho
        self.epsilon = epsilon

        self.cache = {}
        self.delta = {}

    def update(self, fn, param, param_grad, cur_loss=None):
        lr = self.scheduler(self.cur_step, cur_loss)
        cache = self.cache
        delta = self.delta
        rho = self.rho
        eps = self.epsilon

        for p, g in zip(param, param_grad):
            c = cache.get(p)
            if c is None:
                c = np.multiply(1.0 - rho, np.square(g))
            else:
                c = np.add(np.multiply(rho, c), np.multiply(1.0 - rho, np.square(g)))

            d = delta.get(p)
            if d is None:
                d = np.zeros_like(g)

            upd = np.divide(np.multiply(g, np.sqrt(np.add(d, eps))), np.sqrt(np.add(c, eps)))
            fn(p, np.multiply(lr, upd))
            d = np.add(np.multiply(rho, d), np.multiply(1.0 - rho, np.square(upd)))
            cache[p] = c
            delta[p] = d

        """
        # init cache and delta
        if self.cache is None:
            self.cache = [_zero(p.shape) for p in params]
        if self.delta is None:
            self.delta = [_zero(p.shape) for p in params]

        # update parameters
        for i, (c, d, p, g) in enumerate(zip(self.cache, self.delta, params, grads)):
            c = self.rho * c + (1 - self.rho) * np.power(g, 2)
            update = g * np.sqrt(d + self.epsilon) / np.sqrt(c + self.epsilon)
            p -= self.lr * update
            d = self.rho * d + (1 - self.rho) * np.power(update, 2)

            self.cache[i] = c
            self.delta[i] = d
        """
