import numpy as np
import math
import matlab.engine
from functools import partial
import elfi
# # ground-true params
# om = 0.3
# ok = 0
# w0 = -1
# wa = 0
# h0 = 0.7

# log_om = math.log(om)
# log_h0 = math.log(h0)

# thetas = [log_om, ok, w0, wa, log_h0]

# n = 10e+4
# n_bin = 20
# num_sim = 100


def astro_sim(h0, om, w0, n_obs=10e+4, batch_size=1, random_state=None):
    """Actually sim summaries"""
    if hasattr(om, '__len__'):
        om = om[0]
        w0 = w0[0]
        h0 = h0[0]
    print('h0', h0)
    print('om', om)
    print('w0', w0)
    if om <= 0:
        om = 0.001
    if h0 <= 0:
        h0 = 0.001
    log_om = math.log(om)
    log_h0 = math.log(h0)
    ok = 0
    wa = 1 - ok
    thetas = [log_om, ok, w0, wa, log_h0]

    n_bin = 20

    eng = matlab.engine.start_matlab()  # slow!

    thetas = matlab.double(thetas)
    batch_size = int(batch_size)
    sim_summ = eng.astroSL_simsummaries(thetas, n_obs, n_bin, batch_size)
    sim_summ = np.asanyarray(sim_summ).reshape((batch_size, -1))
    eng.quit()
    # print('sim_summ_mat', sim_summ)
    return sim_summ


def identity(x):
    return x


def get_model(n_obs=10e+4, true_params=None, seed_obs=None):
    om = 0.3
    ok = 0
    w0 = -1
    wa = 0
    h0 = 0.7

    if true_params is None:
        true_params = [h0, om, w0]  #om, 
        # true_params = [0.3, 0., -1.0, 0.0, 0.7]

    y = astro_sim(*true_params, random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(astro_sim, n_obs=n_obs)

    m = elfi.ElfiModel()
    priors = []
    # work under flat universe assumption Omega_k = 0
    priors.append(elfi.Prior('uniform', -1, 3, model=m, name="h0"))
    priors.append(elfi.Prior('beta', 3, 3, model=m, name='om'))
    priors.append(elfi.Prior('normal', -1, 0.5, model=m, name="w0"))
    # elfi.Prior('uniform', -1, 2, name='ok')
    # priors.append(elfi.Prior('uniform', 0, 1 - m['om'], name="wa"))  # this is just 1-om if flat
    elfi.Simulator(sim_fn, *priors, observed=y, name='astro')
    elfi.Summary(identity, m['astro'], name='identity')
    return m
