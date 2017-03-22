import pytest

import numpy as np
import scipy.stats as ss

import elfi
import elfi.client


def test_batch_client(simple_model, client):
    # This is an indirect test checking the outcome rather than the actual random_state.
    # One could find the random_state node from the loaded_net and check its state.

    m = simple_model
    context = elfi.ComputationContext(seed=123, batch_size=10)
    client = elfi.client.BatchClient(m.source_net, 'k2', context, client)

    client.submit_batch(0)
    out0, i0 = client.wait_next_batch()

    client.submit_batch(1)
    out1, i1 = client.wait_next_batch()

    client.submit_batch(0)
    out0_, i0_ = client.wait_next_batch()

    assert i0 == 0
    assert i1 == 1
    assert i0_ == 0
    assert np.array_equal(out0['k2'], out0_['k2'])
    assert not np.array_equal(out0['k2'], out1['k2'])
