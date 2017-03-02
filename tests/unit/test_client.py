import pytest

import numpy as np
import scipy.stats as ss

import elfi
from elfi.native_client import Client


def test_batch_queue(simple_model):
    # This is an indirect test checking the outcome rather than the actual random_state.
    # One could find the random_state node from the loaded_net and check its state.

    compiled_net = Client.compile(simple_model.source_net, 'k2')
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