import math

class Scheduler(object):

    def __init__(self, unit_length=1, last_step=-1):
        self.unit_length = unit_length
        self.last_step = last_step

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def get_value(self):
        raise NotImplementedError

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        return self.get_value()


class WarmupScheduler(Scheduler):

    def __init__(self, base_value, warmup_steps=0, warmup_power=1, warmup_value=0, **kwargs):
        super(WarmupScheduler, self).__init__(**kwargs)
        self.base_value = base_value
        self.warmup_steps = warmup_steps * self.unit_length
        self.warmup_power = warmup_power
        self.warmup_value = warmup_value

    def get_value(self):
        if self.last_step < self.warmup_steps:
            progress = self.last_step / self.warmup_steps
            factor = progress ** self.warmup_power
            value = factor * (self.base_value - self.warmup_value) + self.warmup_value
        else:
            value = self.get_value_after_warmup()
        return value

    def get_value_after_warmup(self):
        raise NotImplementedError


class CosineScheduler(WarmupScheduler):

    def __init__(self, base_value, total_steps, min_value=0, **kwargs):
        super(CosineScheduler, self).__init__(base_value, **kwargs)
        self.total_steps = total_steps * self.unit_length
        self.min_value = min_value

    def get_value_after_warmup(self):
        progress = (self.last_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        factor = (math.cos(math.pi * progress) + 1) / 2
        return max(self.base_value * factor, self.min_value)
