import pytest

import numpy as np
import scipy.stats as ss

import elfi
import elfi.client

@pytest.mark.usefixtures('with_all_clients')
def test_batch_handler(simple_model):

    m = simple_model
    m.computation_context = elfi.ComputationContext(seed=123, batch_size=10)
    batches = elfi.client.BatchHandler(m, 'k2')

    batches.submit(0)
    out0, i0 = batches.wait_next()

    batches.submit(1)
    out1, i1 = batches.wait_next()

    batches.submit(0)
    out0_, i0_ = batches.wait_next()

    assert i0 == 0
    assert i1 == 1
    assert i0_ == 0
    assert np.array_equal(out0['k2'], out0_['k2'])
    assert not np.array_equal(out0['k2'], out1['k2'])


# TODO: add testing that client is cleared from tasks after they are retrieved
