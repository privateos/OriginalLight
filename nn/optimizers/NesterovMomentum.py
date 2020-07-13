from .OptimizerBase import OptimizerBase, np

class NesterovMomentum(OptimizerBase):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + momentum * velocity - learning_rate * gradient``

    Parameters
    ----------
    momentum : float
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.

    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.

    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.

    """

    def __init__(self, scheduler, momentum=0.9):
        super(self.__class__, self).__init__(scheduler)

        self.momentum = momentum

        self.velocity = {}

    def update(self, fn, param, param_grad, cur_loss=None):
        lr = self.scheduler(self.cur_step, cur_loss)
        velocity = self.velocicy
        for p, g in zip(param, param_grad):
            v = velocity.get(p)
            v1 = None
            v0 = None
            if v is None:
                v = np.zeros_like(g)
                t = np.multiply(lr, g)
                v0 = np.negative(t)
                v = v0
                v1 = np.subtract(np.multiply(self.momentum, v), t)
            else:  
                t = np.multiply(lr, g)
                v0 = np.subtract(np.multiply(self.momentum, v), t)
                v = v0
                v1 = np.subtract(np.multiply(self.momentum, v), t)
            velocity[p] = v0
            fn(p, np.negative(v1))

        """
        # init the velocities
        if self.velocity is None:
            self.velocity = [_zero(p.shape) for p in params]

        # update the parameters
        for i, (v, p, g) in enumerate(zip(self.velocity, params, grads)):
            v = self.momentum * v - self.lr * g
            p += (self.momentum * v - self.lr * g)
            self.velocity[i] = v

        super(NesterovMomentum, self).update(params, grads)
        """