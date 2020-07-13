from .OptimizerBase import OptimizerBase, np
from math import pow
class Adamax(OptimizerBase):
    """Adamax updates

    Adamax updates implemented as in [1]_. This is a variant of of the Adam
    algorithm based on the infinity norm.

    Parameters
    ----------
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Constant for numerical stability.

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
        ms = self.ms
        vs = self.vs
        beta2 = self.beta2
        beta1 = self.beta1
        eps = self.epsilon

        self.iterations += 1
        a_t = lr / (1.0 - pow(beta1, self.iterations))

        for p, g in zip(param, param_grad):
            m = ms.get(p)
            if m is None:
                m = np.zeros_like(g)
            v = vs.get(p)
            if v is None:
                v = np.zeros_like(g)
            fn(p, np.divide(np.multiply(a_t, m), np.add(eps, v)))
            self.ms[p] = m
            self.vs[p] = v
            
        """
        # init
        self.iterations += 1
        a_t = self.lr / (1 - np.power(self.beta1, self.iterations))
        if self.ms is None:
            self.ms = [_zero(p.shape) for p in params]
        if self.vs is None:
            self.vs = [_zero(p.shape) for p in params]

        # update parameters
        for i, (m, v, p, g) in enumerate(zip(self.ms, self.vs, params, grads)):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = np.maximum(self.beta2 * v, np.abs(g))
            p -= a_t * m / (v + self.epsilon)

            self.ms[i] = m
            self.vs[i] = v
        """
