from .OptimizerBase import OptimizerBase, np

class SGD(OptimizerBase):
    def __init__(self, scheduler):
        super(self.__class__, self).__init__(scheduler)
    
    def update(self, fn,  param, param_grad, cur_loss=None):
        lr = self.scheduler(self.cur_step, cur_loss)
        for p, g in zip(param, param_grad):
            fn(p, np.multiply(lr, g))
    
        