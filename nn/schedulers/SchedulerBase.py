from light.backend import backend as np
class SchedulerBase(object):
    def __init__(self):
        self.hyperparameters = {}

    def __call__(self, step=None, cur_loss=None):
        return self.learning_rate(step=step, cur_loss=cur_loss)

    def set_params(self, hparam_dict):
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v
        
    def learning_rate(self, **kwargs):
        raise NotImplementedError

