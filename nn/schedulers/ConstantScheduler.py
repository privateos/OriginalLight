from .SchedulerBase import SchedulerBase, np
class ConstantScheduler(SchedulerBase):
    def __init__(self, lr=0.01, **kwargs):
        super(self.__class__, self).__init__()
        self.lr = lr
        self.hyperparameters = {"id":"ConstantScheduler", "lr":self.lr}

    def __str__(self):
        return 'ConstantScheduler(lr={})'.format(self.lr)

    def learning_rate(self, **kwargs):
        return self.lr
