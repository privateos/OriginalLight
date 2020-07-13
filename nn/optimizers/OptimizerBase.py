from light.backend import backend as np
from light.nn.schedulers import ConstantScheduler
class OptimizerBase(object):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.cur_step = 0
    
    #fn:用来改变梯度的函数
    def __call__(self, fn, param, param_grad, cur_loss=None):
        self.update(fn, param, param_grad, cur_loss)
    
    def step(self):
        self.cur_step += 1
    
    def reset_step(self):
        self.cur_step = 0

    def update(self, fn, param, param_grad, cur_loss=None):
        raise NotImplementedError
    