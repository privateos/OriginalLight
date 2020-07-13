from .SchedulerBase import SchedulerBase, np
class NoamScheduler(SchedulerBase):
    def __init__(self, model_dim=512, scale_factor=1.0, warmup_steps=4000, **kwargs):
        """
        The Noam learning rate scheduler, originally used in conjunction with
        the Adam optimizer in [1].

        Notes
        -----
        The Noam scheduler increases the learning rate linearly for the first
        `warmup_steps` steps, and decreases it thereafter proportionally to the
        inverse square root of the step number::

            lr = scale_factor * ( (model_dim ** (-0.5)) * adj_step )
            adj_step = min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

        References
        ----------
        .. [1] Vaswani et al. (2017) "Attention is all you need". *31st
           Conference on Neural Information Processing Systems*,
           https://arxiv.org/pdf/1706.03762.pdf

        Parameters
        ----------
        model_dim : int
            The number of units in the layer output. Default is 512.
        scale_factor : float
            A fixed coefficient for rescaling the final learning rate. Default
            is 1.
        warmup_steps : int
            The number of steps in the warmup stage of training. Default is
            4000.
        """
        super(self.__class__, self).__init__()
        self.model_dim = model_dim
        self.scale_factor = scale_factor
        self.warmup_steps = warmup_steps
        self.hyperparameters = {
            'id':'NoamScheduler',
            'model_dim':self.model_dim,
            'scale_factor':self.scale_factor,
            'warmup_steps':self.warmup_steps
        }

    def __str__(self):
        return "NoamScheduler(model_dim={}, scale_factor={}, warmup_steps={})".format(
            self.model_dim, self.scale_factor, self.warmup_steps
        )
    
    def learning_rate(self, step, **kwargs):
        warmup, d_model = self.warmup_steps, self.model_dim
        new_lr = d_model**(-0.5)*min(step**(-0.5), step*warmup**(-1.5))
        return self.scale_factor*new_lr
