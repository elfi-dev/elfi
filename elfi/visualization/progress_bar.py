from IPython.display import display
from ipywidgets import HTML, FloatProgress, VBox


class ProgressBar(object):

    """Progress bar for showing the inference process."""

    def __init__(self, batch_size, n_samples):
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.step = self.n_samples / self.batch_size
        self.n_iter = self.step * 100
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
