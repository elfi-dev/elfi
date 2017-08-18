import numpy as np
import pytest

import elfi
from elfi.utils import is_array


def simulator(p, random_state=None):
    n = 30
    rs = random_state or np.random.RandomState()
    data = rs.multinomial(n, p)

    # Make it a dict for testing purposes
    return dict(zip(range(n), data))


def summary(dict_data):
    n = len(dict_data)
    data = np.array([dict_data[i] for i in range(n)])
    return data / n


def lsimulator(p, random_state=None):
    n = 30
    rs = random_state or np.random.RandomState()
    data = rs.multinomial(n, p)

    # Make it a list for testing purposes
    return list(data) + ['test']


def lsummary(list_data):
    n = len(list_data)
    data = np.array(list_data[:-1])
    return data / (n - 1)


def test_dict_output():
    vsim = elfi.tools.vectorize(simulator)
    vsum = elfi.tools.vectorize(summary)

    obs = simulator([.2, .8])

    elfi.new_model()
    p = elfi.Prior('dirichlet', [2, 2])
    sim = elfi.Simulator(vsim, p, observed=obs)
    S = elfi.Summary(vsum, sim)
    d = elfi.Distance('euclidean', S)

    pool = elfi.OutputPool(['sim'])
    rej = elfi.Rejection(d, batch_size=100, pool=pool, output_names=['sim'])
    sample = rej.sample(100, n_sim=1000)
    mean = np.mean(sample.samples['p'], axis=0)

    # Crude test
    assert mean[1] > mean[0]


def test_list_output():
    vsim = elfi.tools.vectorize(lsimulator)
    vsum = elfi.tools.vectorize(lsummary)

    v = vsim(np.array([[.2, .8], [.3, .7]]))
    assert is_array(v)
    assert not isinstance(v[0], list)

    vsim = elfi.tools.vectorize(lsimulator, dtype=False)

    v = vsim(np.array([[.2, .8], [.3, .7]]))
    assert is_array(v)
    assert isinstance(v[0], list)

    obs = lsimulator([.2, .8])

    elfi.new_model()
    p = elfi.Prior('dirichlet', [2, 2])
    sim = elfi.Simulator(vsim, p, observed=obs)
    S = elfi.Summary(vsum, sim)
    d = elfi.Distance('euclidean', S)

    pool = elfi.OutputPool(['sim'])
    rej = elfi.Rejection(d, batch_size=100, pool=pool, output_names=['sim'])
    sample = rej.sample(100, n_sim=1000)
    mean = np.mean(sample.samples['p'], axis=0)

    # Crude test
    assert mean[1] > mean[0]
