import pytest

import numpy as np
import scipy.stats as ss

import elfi


def test_batch_queue(simple_model, client):
    # This is an indirect test checking the outcome rather than the actual random_state.
    # One could find the random_state node from the loaded_net and check its state.

    compiled_net = client.compile(simple_model.source_net, 'k2')
    context = elfi.ComputationContext(seed=123, batch_size=10)

    client.submit_batches([0, 1, 0], compiled_net, context)

    out0, i0 = client.wait_next_batch()
    out1, i1 = client.wait_next_batch()
    out0_, i0_ = client.wait_next_batch()

    assert i0 == 0
    assert i1 == 1
    assert i0_ == 0
    assert np.array_equal(out0['k2'], out0_['k2'])
    assert not np.array_equal(out0['k2'], out1['k2'])
