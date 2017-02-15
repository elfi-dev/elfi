import pytest

import numpy as np
import scipy.stats as ss

import elfi


def model():
    tau = elfi.Constant('tau', 10)
    k1 = elfi.Prior('k1', 'uniform', 0, tau, size=1)
    k2 = elfi.Prior('k2', 'normal', k1, size=3)
    return k2.network


def random_get_state_equal(st1, st2):
    # 1. the string 'MT19937'.
    tf = st1[0] == st2[0]
    # 2. a 1-D array of 624 unsigned integer keys.
    tf = tf and np.array_equal(st1[1], st2[1])
    # 3. an integer ``pos``.
    tf = tf and st1[2] == st2[2]
    return tf


def test_randomness():
    m = model()
    k1 = m.get_reference('k1')

    output1 = k1.generate(10)
    output2 = k1.generate(10)

    assert not np.array_equal(output1, output2)


def test_global_random_state_usage():
    n_gen = 10

    m = model()
    np.random.seed(0)
    k2 = m.get_reference('k2')
    k2.generate(n_gen)
    st1 = np.random.get_state()

    np.random.seed(0)
    mu = ss.uniform.rvs(0, 10, size=(n_gen, 1))
    ss.norm.rvs(mu, size=(n_gen, 3))
    st2 = np.random.get_state()

    assert random_get_state_equal(st1, st2)
