from .SchedulerBase import SchedulerBase, np
class ExponentialScheduler(SchedulerBase):
    def __init__(self, initial_lr=0.01, stage_length=500, staircase=False, decay=0.1):
        """
        An exponential learning rate scheduler.

        Notes
        -----
        The exponential scheduler decays the learning rate by `decay` every
        `stage_length` steps, starting from `initial_lr`::

            learning_rate = initial_lr * decay ** curr_stage

        where::

            curr_stage = step / stage_length          if staircase = False
            curr_stage = floor(step / stage_length)   if staircase = True

        Parameters
        ----------
        initial_lr : float
            The learning rate at the first step. Default is 0.9.
        stage_length : int
            The length of each stage, in steps. Default is 500.
        staircase : bool
            If True, only adjusts the learning rate at the stage transitions,
            producing a step-like decay schedule. If False, adjusts the
            learning rate after each step, creating a smooth decay schedule.
            Default is False.
        decay : float
            The amount to decay the learning rate at each new stage. Default is
            0.1.
        """
        super(self.__class__, self).__init__()
        self.decay = decay
        self.staircase = staircase
        self.initial_lr = initial_lr
        self.stage_length = stage_length
        self.hyperparameters = {
            'id':'StepScheduler',
            'decay':self.decay,
            'staircase':self.staircase,
            'initial_lr':self.initial_lr,
            'stage_length':self.stage_length
        }

    def __str__(self):
        return "ExponentialScheduler(initial_lr={}, stage_length={}, staircase={}, decay={})".format(
            self.initial_lr, self.stage_length, self.staircase, self.decay
        )

    def learning_rate(self, step, **kwargs):
        cur_stage = step/self.stage_length
        if self.staircase:
            cur_stage = np.floor(cur_stage)
        return self.initial_lr* (self.decay**cur_stage)

