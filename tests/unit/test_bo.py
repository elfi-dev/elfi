import pytest

import numpy as np

import elfi


@pytest.mark.usefixtures('with_all_clients')
def test_BO(ma2):
    # Log transform of the distance usually smooths the distance surface
    log_d = elfi.Operation(np.log, ma2['d'], name='log_d')

    n_init = 20
    res_init = elfi.Rejection(log_d, batch_size=5).sample(n_init, quantile=1)

    bounds = {n:(-2, 2) for n in ma2.parameter_names}
    bo = elfi.BayesianOptimization(log_d, initial_evidence=res_init.outputs,
                                   update_interval=10, batch_size=5,
                                   bounds=bounds)
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


def test_acquisition():
    n_params = 2
    n = 10
    n2 = 5
    parameter_names = ['a', 'b']
    bounds = {'a':[-2, 3], 'b':[5, 6]}
    target_model = elfi.methods.bo.gpy_regression.GPyRegression(parameter_names, bounds=bounds)
    x1 = np.random.uniform(*bounds['a'], n)
    x2 = np.random.uniform(*bounds['b'], n)
    x = np.column_stack((x1, x2))
    y = np.random.rand(n)
    target_model.update(x, y)

    # check acquisition without noise
    acq_noise_cov = 0
    t = 1
    acquisition_method = elfi.methods.bo.acquisition.LCBSC(target_model, noise_cov=acq_noise_cov)
    new = acquisition_method.acquire(n2, t=t)
    assert np.allclose(new[1:, 0], new[0, 0])
    assert np.allclose(new[1:, 1], new[0, 1])

    # check acquisition with scalar noise
    acq_noise_cov = 2
    t = 1
    acquisition_method = elfi.methods.bo.acquisition.LCBSC(target_model, noise_cov=acq_noise_cov)
    new = acquisition_method.acquire(n2, t=t)
    assert new.shape == (n2, n_params)
    assert np.all((new[:, 0] >= bounds['a'][0]) & (new[:, 0] <= bounds['a'][1]))
    assert np.all((new[:, 1] >= bounds['b'][0]) & (new[:, 1] <= bounds['b'][1]))

    # check acquisition with diagonal covariance
    acq_noise_cov = np.random.uniform(0, 5, size=2)
    t = 1
    acquisition_method = elfi.methods.bo.acquisition.LCBSC(target_model, noise_cov=acq_noise_cov)
    new = acquisition_method.acquire(n2, t=t)
    assert new.shape == (n2, n_params)
    assert np.all((new[:, 0] >= bounds['a'][0]) & (new[:, 0] <= bounds['a'][1]))
    assert np.all((new[:, 1] >= bounds['b'][0]) & (new[:, 1] <= bounds['b'][1]))

    # check acquisition with arbitrary covariance matrix
    acq_noise_cov = np.random.rand(n_params, n_params) * 0.5
    acq_noise_cov += acq_noise_cov.T
    acq_noise_cov += n_params * np.eye(n_params)
    t = 1
    acquisition_method = elfi.methods.bo.acquisition.LCBSC(target_model, noise_cov=acq_noise_cov)
    new = acquisition_method.acquire(n2, t=t)
    assert new.shape == (n2, n_params)
    assert np.all((new[:, 0] >= bounds['a'][0]) & (new[:, 0] <= bounds['a'][1]))
    assert np.all((new[:, 1] >= bounds['b'][0]) & (new[:, 1] <= bounds['b'][1]))

    # test Uniform Acquisition
    t = 1
    acquisition_method = elfi.methods.bo.acquisition.UniformAcquisition(target_model, noise_cov=acq_noise_cov)
    new = acquisition_method.acquire(n2, t=t)
    assert new.shape == (n2, n_params)
    assert np.all((new[:, 0] >= bounds['a'][0]) & (new[:, 0] <= bounds['a'][1]))
    assert np.all((new[:, 1] >= bounds['b'][0]) & (new[:, 1] <= bounds['b'][1]))
