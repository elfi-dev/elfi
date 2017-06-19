import pytest

import numpy as np

import elfi


@pytest.mark.usefixtures('with_all_clients')
def test_BO(ma2):
    # Log transform of the distance usually smooths the distance surface
    log_d = elfi.Operation(np.log, ma2['d'], name='log_d')

    n_init = 20
    res_init = elfi.Rejection(log_d, batch_size=5).sample(n_init, quantile=1)

    bo = elfi.BayesianOptimization(log_d, initial_evidence=res_init.outputs,
                                   update_interval=10, batch_size=5,
                                   bounds=[(-2,2)]*len(ma2.parameter_names))
    assert bo.target_model.n_evidence == n_init
    assert bo.n_evidence == n_init
    assert bo._n_precomputed == n_init
    assert bo.n_initial_evidence == n_init

    n1 = 5
    bo.infer(n_init + n1)

    assert bo.target_model.n_evidence == n_init + n1
    assert bo.n_evidence == n_init + n1
    assert bo._n_precomputed == n_init
    assert bo.n_initial_evidence == n_init

    n2 = 5
    bo.infer(n_init + n1 + n2)

    assert bo.target_model.n_evidence == n_init + n1 + n2
    assert bo.n_evidence == n_init + n1 + n2
    assert bo._n_precomputed == n_init
    assert bo.n_initial_evidence == n_init

    assert np.array_equal(bo.target_model._gp.X[:n_init, 0], res_init.samples_list[0])
