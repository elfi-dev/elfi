import pytest

import numpy as np
import scipy.stats as ss

import elfi
from elfi.native_client import Client

def model(computation_context=None):
    m = elfi.ElfiModel(computation_context=computation_context)
    tau = elfi.Constant('tau', 10, model=m)
    k1 = elfi.Prior('k1', 'uniform', 0, tau, size=1, model=m)
    k2 = elfi.Prior('k2', 'normal', k1, size=3, model=m)
    return m


def random_state_equal(st1, st2):
    # 1. the string 'MT19937'.
    tf = st1[0] == st2[0]
    # 2. a 1-D array of 624 unsigned integer keys.
    tf = tf and np.array_equal(st1[1], st2[1])
    # 3. an integer ``pos``.
    tf = tf and st1[2] == st2[2]
    return tf


def test_randomness():
    m = model()
    k1 = m['k1']

    gen1 = k1.generate(10)
    gen2 = k1.generate(10)

    assert not np.array_equal(gen1, gen2)


def test_global_random_state_usage():
    n_gen = 10

    m = model()
    np.random.seed(0)
    k2 = m['k2']
    k2.generate(n_gen)
    st1 = np.random.get_state()

    np.random.seed(0)
    mu = ss.uniform.rvs(0, 10, size=(n_gen, 1))
    ss.norm.rvs(mu, size=(n_gen, 3))
    st2 = np.random.get_state()

    assert random_state_equal(st1, st2)


def test_consistency_with_a_seed():
    context = elfi.ComputationContext(seed=123)
    m = model(context)
    gen1 = m['k2'].generate(10)

    context = elfi.ComputationContext(seed=123)
    m = model(context)
    gen2 = m['k2'].generate(10)

    assert np.array_equal(gen1, gen2)


def test_different_states_for_different_batches():
    # This is an indirect test checking the outcome rather than the actual random_state.
    # One could find the random_state node from the loaded_net and check its state.

    m = model()
    compiled_net = Client.compile(m.source_net, 'k2')

    context = elfi.ComputationContext(seed=123, batch_size=10)
    Client.submit_batches([0, 0, 1], compiled_net, context)

    out0, i0 = Client.wait_next_batch()
    out0_, i0_ = Client.wait_next_batch()
    out1, i1 = Client.wait_next_batch()

    assert i0 == 0
    assert i0_ == 0
    assert i1 == 1
    assert np.array_equal(out0['k2'], out0_['k2'])
    assert not np.array_equal(out0['k2'], out1['k2'])
