import numpy as np
import pytest
import scipy.stats as ss

import elfi
import elfi.client


@pytest.mark.usefixtures('with_all_clients')
def test_batch_handler(simple_model):
    m = simple_model
    computation_context = elfi.ComputationContext(seed=123, batch_size=10)
    batches = elfi.client.BatchHandler(m, computation_context, 'k2')

    batches.submit()
    out0, i0 = batches.wait_next()

    batches.submit()
    out1, i1 = batches.wait_next()

    batches.reset()
    batches.submit()
    out0_, i0_ = batches.wait_next()

    assert i0 == 0
    assert i1 == 1
    assert i0_ == 0
    assert np.array_equal(out0['k2'], out0_['k2'])
    assert not np.array_equal(out0['k2'], out1['k2'])


def test_multiprocessing_kwargs(simple_model):
    pre = elfi.get_client()

    m = simple_model
    num_proc = 2
    elfi.set_client('multiprocessing', num_processes=num_proc)
    rej = elfi.Rejection(m['k1'])
    assert rej.client.num_cores == num_proc

    elfi.set_client('multiprocessing', processes=num_proc)
    rej = elfi.Rejection(m['k1'])
    assert rej.client.num_cores == num_proc

    elfi.set_client(pre)

# TODO: add testing that client is cleared from tasks after they are retrieved
