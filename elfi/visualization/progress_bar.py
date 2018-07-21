"""Progress bar for inference methods."""

from math import ceil

from IPython.display import display
from ipywidgets import HTML, IntProgress, VBox


class ProgressBar(object):
    """Progress bar for showing the inference process."""

    def __init__(self, batch_size, n_samples, sampler, quantile=None, n_sim=None):
        """Construct the progress bar.

        Parameters
        ----------
        batch_size : int, required
        n_samples : int, required
            Number of samples to generate from the (approximate) posterior
        sampler : str, required
            The name of inference method
        quantile : optional
            In between (0,1). Define the threshold as the p-quantile of all the
            simulations. n_sim = n_samples/quantile.
        n_sim : int, optional
            Number of the iterations of the specific sampling.

        """
        sampler = sampler.lower()
        if sampler in ['rejection', 'sampler']:
            if n_sim and not quantile:
                self.n_iter = n_sim
                self.step = ceil(self.n_iter / 100)
            if quantile and n_sim:
                self.n_iter = ceil(n_samples / quantile)
                self.step = ceil(self.n_iter / 100)
        elif sampler == 'bolfi':
            self.n_iter = n_sim
            self.step = ceil(n_sim / 100)
        else:
            self.n_iter = n_sim if n_sim else n_samples / batch_size
            self.step = self.n_iter / 100
        self.bar = IntProgress(min=0, max=self.n_iter)
        self.label = HTML()
        self.box = VBox(children=[self.label, self.bar])
        display(self.box)

    def update(self):
        """Update the step in progress bar."""
        self.bar.value += self.step
        self.label.value = '{name}: {index} / {size}'.format(
            name="Inference progress bar (number of simulations)",
            index=self.bar.value,
            size=self.n_iter,
        )
