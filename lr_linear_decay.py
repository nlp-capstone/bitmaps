class LRLinearDecay:
    def __init__(self, warmup, total):
        self.warmup = warmup
        self.total = total

    def get_lr_fn(self):
        return lambda step: self.get_lr(step)

    def get_lr(self, step):
        if step < self.warmup:
            return step / self.warmup

        if step >= self.total:
            return 0.0

        return (self.total - step) / (self.total - self.warmup)
