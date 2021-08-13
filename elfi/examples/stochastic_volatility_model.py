from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def shock_term(alpha, beta, kappa, eta, batch_size=1):
    v_t = ss.levy_stable.rvs(alpha=alpha,
                         beta=beta,
                         loc=eta,
                         scale=kappa,
                         size=batch_size)
    return v_t


def alpha_stochastic_volatility_model(alpha,
                                      beta,
                                      n_obs=50,
                                      batch_size=1,
                                      random_state=None):
    random_state = random_state or np.random

    # assumes remaining parameters are known and fixed
    mu = 5
    phi = 1
    kappa = 1
    eta = 0
    sigma = 0.2

    # draw log volatility term (x_t)
    y_mat = np.zeros((batch_size, n_obs))
    # first time step (does not rely on prev xx_t)
    # x_0 = np.random.multivariate_normal(mu+phi*(np.zeros(batch_size)-mu), sigma*np.eye(batch_size))
    x_0 = np.random.normal(mu+phi*-mu, sigma, batch_size)
    v_0 = shock_term(alpha, beta, kappa, eta, batch_size)
    y_mat[:, 0] = x_0*v_0  # assumes x_0 has no prev.
    x_prev = x_0
    for t in range(1, n_obs):
        x_t = [np.random.normal(mu+phi*(x_prev[i]-mu), sigma) for i in range(batch_size)]
        v_t = shock_term(alpha, beta, kappa, eta, batch_size)
        y_mat[:, t] = x_t * v_t
        x_prev = x_t

    return y_mat
    # draw shock term from stable distribution


def identity_fun(x):
    return x


def get_model(n_obs=50, true_params=None):
    if true_params is None:
        true_params = [1.2, 0.5]

    m = elfi.ElfiModel()
    simulator = partial(alpha_stochastic_volatility_model, n_obs=n_obs)
    y_obs = simulator(*true_params)
    elfi.Prior('uniform', 0, 2, model=m, name='alpha')
    elfi.Prior('uniform', 0, 1, model=m, name='beta')
    elfi.Simulator(alpha_stochastic_volatility_model, m['alpha'], m['beta'], observed=y_obs, name='a_svm')
    elfi.Summary(identity_fun,  m['a_svm'], name="identity")

    return m
