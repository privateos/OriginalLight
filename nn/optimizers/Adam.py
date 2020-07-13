from .OptimizerBase import OptimizerBase, np
from math import pow, sqrt
class Adam(OptimizerBase):
    """Adam updates

    Adam updates implemented as in [1]_.

    Parameters
    ----------
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Constant for numerical stability.

    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """

    def __init__(self, scheduler, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(self.__class__, self).__init__(scheduler)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.ms = {}
        self.vs = {}
        self.iterations = 0

    def update(self, fn, param, param_grad, cur_loss=None):
        lr = self.scheduler(self.cur_step, cur_loss)
        beta1 = self.beta1
        beta2 = self.beta2
        ms = self.ms
        vs = self.vs
        eps = self.epsilon

        self.iterations += 1
        a_t = lr * sqrt(1.0 - pow(beta2, self.iterations)) / (1.0 - pow(beta1, self.iterations))

        for p, g in zip(param, param_grad):
            m = ms.get(p)
            if m is None:
                m = np.zeros_like(g)
            v = vs.get(p)
            if v is None:
                v = np.zeros_like(g)
            m = np.add(np.multiply(beta1, m), np.multiply(1.0 - beta1, g))
            v = np.add(np.multiply(beta2, v), np.multiply(1.0 - beta2, np.square(g)))
            fn(p, np.divide(np.multiply(a_t, m), np.add(eps, np.sqrt(v))))
            ms[p] = m
            vs[p] = v

        """
        # init
        self.iterations += 1
        a_t = self.lr * np.sqrt(1 - np.power(self.beta2, self.iterations)) / \
              (1 - np.power(self.beta1, self.iterations))
        if self.ms is None:
            self.ms = [_zero(p.shape) for p in params]
        if self.vs is None:
            self.vs = [_zero(p.shape) for p in params]

        # update parameters
        for i, (m, v, p, g) in enumerate(zip(self.ms, self.vs, params, grads)):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * np.power(g, 2)
            p -= a_t * m / (np.sqrt(v) + self.epsilon)

            self.ms[i] = m
            self.vs[i] = v
        """
