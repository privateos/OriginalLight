from .OptimizerBase import OptimizerBase, np

class Momentum(OptimizerBase):
    """Stochastic Gradient Descent (SGD) updates with momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + velocity``

    Parameters
    ----------
    momentum : float
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    """

    def __init__(self, scheduler, momentum=0.9,):
        super(self.__calss__, self).__init__(scheduler)

        self.momentum = momentum

        self.velocity = {}

    def update(self, fn, param, param_grad, cur_loss=None):
        lr = self.scheduler(self.cur_step, cur_loss)
        velocity = self.velocicy
        
        for p, g in zip(param, param_grad):
            v = velocity.get(p)
            v0 = None
            if v is None:
                v0 = np.negative(np.multiply(lr, g))
            else:
                v0 = np.subtract(np.multiply(self.momentum, v), np.multiply(lr, g))

            fn(p, np.negative(v0))
            velocity[p] = v0
        """
        if self.velocity is None:
            self.velocity = [_zero(p.shape) for p in params]

        # update the parameters
        for i, (v, p, g) in enumerate(zip(self.velocity, params, grads)):
            v = self.momentum * v - self.lr * g
            p += v
            self.velocity[i] = v

        """