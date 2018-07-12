from math import ceil

from IPython.display import display
from ipywidgets import HTML, FloatProgress, VBox


class ProgressBar(object):
    """Progress bar for showing the inference process."""

    def __init__(self, batch_size, n_samples, sampler, n_sim=None, threshold=None):
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.sampler = sampler
        self.n_sim = n_sim
        self.threshold = threshold
        if self.sampler == 'rejection':
            if n_sim:
                self.n_iter = ceil(self.n_sim / self.batch_size)
                self.step = ceil(self.n_iter / 100)
            if not n_sim and threshold:
                self.n_iter = ceil(self.n_samples / self.batch_size)
                self.step = ceil(self.n_iter / 100)
        if self.sampler == 'smc':
            if threshold and isinstance(threshold, (list, tuple)):
                self.n_iter = ceil(self.n_samples / self.batch_size)
                self.step = ceil(self.n_iter / 100)
        if self.sampler == 'bayesian':
            pass
        if self.sampler == 'bolfi':
            pass
        self.bar = FloatProgress(min=0, max=self.n_iter)
        self.label = HTML()
        self.box = VBox(children=[self.label, self.bar])
        display(self.box)

    def update(self):
        self.bar.value += self.step
        self.label.value = '{name}: {index} / {size}'.format(
            name="Inference progress bar",
            index=self.bar.value,
            size=self.n_iter,
        )
