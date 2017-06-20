import elfi

import numpy as np


def simulator(p, random_state=None):
    n = 30
    rs = random_state or np.random.RandomState()
    data = rs.multinomial(n, p)

    # Make it a dict for testing purposes
    return dict(zip(range(n), data))


def summary(dict_data):
    n = len(dict_data)
    data = np.array([dict_data[i] for i in range(n)])
    return data/n


def test_dict_output():
    vsim = elfi.tools.vectorize(simulator)
    vsum = elfi.tools.vectorize(summary)

    obs = simulator([.2, .8])

    elfi.ElfiModel()
    p = elfi.Prior('dirichlet', [2, 2])
    sim = elfi.Simulator(vsim, p, observed=obs)
    S = elfi.Summary(vsum, sim)
    d = elfi.Distance('euclidean', S)

    pool = elfi.OutputPool(['sim'])
    rej = elfi.Rejection(d, batch_size=100, pool=pool, output_names=['sim'])
    sample = rej.sample(100, n_sim=1000)
    mean = np.mean(sample.samples['p'], axis=0)

    # Crude test
    assert mean[0] < mean[1]
