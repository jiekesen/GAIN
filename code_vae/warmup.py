# src/vae/warmup.py
class DeterministicWarmup:
    """
    Linearly increase beta from 0 to t_max over n steps.
    """
    def __init__(self, n=100, t_max=1.0):
        self.t = 0.0
        self.t_max = float(t_max)
        self.inc = 1.0 / float(n)

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc
        self.t = self.t_max if t > self.t_max else t
        return self.t
