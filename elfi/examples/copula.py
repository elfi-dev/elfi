"""
Toy model from the paper

Jingjing Li, David J. Nott, Yanan Fan, Scott A. Sisson (2016)
Extending approximate Bayesian computation methods to high dimensions
via Gaussian copula.
https://arxiv.org/abs/1504.04093v1
"""

import numpy as np
import scipy.stats as ss

import elfi


class TwistedNormal(object):
    """Essentially a joint distribution of independent normal distributions
    with the exeption of the first two componennts.

    The probability density function is proportional to
    ..math:
      p(\theta) \propto \exp( - \frac{\theta_1^2}{200}
       - \frac{(\theta_2 - b \theta_1^2 + 100b)^2}{2} - \sum_{j=3}^p \theta_j^2)
    """

    def __init__(self, p=2, b=0.1):
        if p < 2:
            raise ValueError("Dimensionality must be at least 2.")

        self.p = p
        self.b = b

    def rvs(self, size=1, **kwargs):
        diagonal = np.ones(self.p)
        diagonal[0] = 100
        if size == 1:
            X = ss.multivariate_normal.rvs(mean=np.zeros(self.p),
                                           cov=np.diag(diagonal), size=size, **kwargs)[np.newaxis, :]
        else:
            X = ss.multivariate_normal.rvs(mean=np.zeros(self.p),
                                           cov=np.diag(diagonal), size=size, **kwargs)

        X[:, 1] = X[:, 1] + self.b*X[:, 0]**2 - 100*self.b
        return X


class Simulator(object):
    """A multivariate normal distribution with uncorrelated components."""

    def __init__(self, p=2, sigma=1):
        if p < 2:
            raise ValueError("Dimensionality must be at least 2.")

        self.p = p
        self.cov = sigma*np.eye(p)

    def __call__(self, mus, batch_size=1, random_state=None):
        if batch_size == 1:
            return ss.multivariate_normal.rvs(mean=mus[0],
                                              cov=self.cov, random_state=random_state)[np.newaxis, :]
        else:
            simulations = [ss.multivariate_normal.rvs(mean=mu,
                                                      cov=self.cov, random_state=random_state)
                           for mu in mus]
            return np.stack(simulations)


def get_model(p):
    y_obs = np.zeros(p)
    y_obs[0] = 10
    y_obs = y_obs[None, :]

    sim = Simulator(p=p)
    m = elfi.ElfiModel(set_current=False)
    mu = elfi.Prior(TwistedNormal(p=p), model=m, name='mu')
    Sim = elfi.Simulator(sim, mu, observed=y_obs, name='Gauss')

    return m
