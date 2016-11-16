import numpy as np
import matplotlib.pyplot as plt

from . import core
from .decorators import to_elfi_distribution


__all__ = ('MA2', 'Normal')


class MA2(core.Simulator):
    """Moving average simulator."""
    def __init__(self, n_obs, t1_act, t2_act, t1=None, t2=None, name='MA2',
                 latents=None, vectorized=True, prng=None, **kwargs):
        self.t1_act = t1_act
        self.t2_act = t2_act

        # Store priors as attributes?
        t1 = to_elfi_distribution(t1) or t1_act
        t2 = to_elfi_distribution(t2) or t2_act

        self.prng = prng
        self.n_obs = n_obs
        self.latents = latents or self._latents()
        # super(MA2).__init__(name=name, simulator=self.simulator,
        #                     vectorized=vectorized, **kwargs)
        # TODO: Fix the api. *args are a very bad idea with super
        super(MA2, self).__init__(name, self.simulator, t1, t2,
                                  vectorized=vectorized,
                                  observed=self._observed(), **kwargs)

    # @staticmethod
    def simulator(self, t1, t2, n_obs=None, n_sim=1, prng=None, latents=None):
        """Run the simulator."""
        n_obs = n_obs or self.n_obs
        prng = prng or np.random.RandomState()
        if latents is None:
            latents = prng.randn(n_sim, n_obs + 2)  # ~ N(0, 1) i.i.d
        u = np.atleast_2d(latents)
        y = u[:, 2:] + t1 * u[:, 1:-1] + t2 * u[:, :-2]
        return y

    def plot(self):
        fig = plt.figure()
        p1 = plt.plot(np.arange(0, self.n_obs), self.observed[0, :])
        p2 = plt.scatter(np.arange(-2, self.n_obs), self.latents)
        return fig

    def __call__(self, t1, t2, n_sim=1):
        return self.simulator(t1=t1, t2=t2, n_sim=n_sim,
                              n_obs=self.n_obs, prng=self.prng,
                              latents=self.latents)

    def _latents(self):
        return np.random.randn(self.n_obs + 2)

    def _observed(self):
        return self(t1=self.t1_act, t2=self.t2_act)


class Normal(core.Simulator):
    def __init__(self, mu_act, sigma_act, mu=None, sigma=None, batch_size=1,
                 prng=None, **kwargs):
        self.mu_act = mu_act
        self.sigma_act = sigma_act

        mu = mu or mu_act
        sigma = sigma or sigma_act

        self.batch_size = batch_size
        self.prng = prng or np.random.RandomState()
        name = 'Normal'
        super(Normal, self).__init__(name, self.simulator, mu, sigma,
                                     observed=self._observed(),
                                     vectorized=True, **kwargs)

    @staticmethod
    def simulator(mu, sigma, batch_size=1, prng=None):
        prng = prng or np.random.RandomState()
        return sigma * prng.rand(batch_size) + mu

    def _observed(self):
        return self.simulator(mu=self.mu_act, sigma=self.sigma_act,
                              batch_size=self.batch_size, prng=self.prng)
