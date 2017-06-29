import pytest

import numpy as np

import elfi
import elfi.methods.bo.acquisition as acquisition
from elfi.methods.bo.gpy_regression import GPyRegression

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
    assert bo.n_precomputed_evidence == n_init
    assert bo.n_initial_evidence == n_init

    n1 = 5
    bo.infer(n_init + n1)

    assert bo.target_model.n_evidence == n_init + n1
    assert bo.n_evidence == n_init + n1
    assert bo.n_precomputed_evidence == n_init
    assert bo.n_initial_evidence == n_init

    n2 = 5
    bo.infer(n_init + n1 + n2)

    assert bo.target_model.n_evidence == n_init + n1 + n2
    assert bo.n_evidence == n_init + n1 + n2
    assert bo.n_precomputed_evidence == n_init
    assert bo.n_initial_evidence == n_init

    assert np.array_equal(bo.target_model._gp.X[:n_init, 0], res_init.samples_list[0])

@pytest.mark.usefixtures('with_all_clients')
def test_BO_works_with_zero_init_samples(ma2):
    log_d = elfi.Operation(np.log, ma2['d'], name='log_d')
    bounds = {n:(-2, 2) for n in ma2.parameter_names}
    bo = elfi.BayesianOptimization(log_d, initial_evidence=0,
                                   update_interval=4, batch_size=2,
                                   bounds=bounds)
    assert bo.target_model.n_evidence == 0
    assert bo.n_evidence == 0
    assert bo._n_precomputed == 0
    assert bo.n_initial_evidence == 0
    samples = 4
    bo.infer(samples)
    assert bo.target_model.n_evidence == samples
    assert bo.n_evidence == samples
    assert bo._n_precomputed == 0
    assert bo.n_initial_evidence == 0

def test_acquisition():
    n_params = 2
    n = 10
    n2 = 5
    parameter_names = ['a', 'b']
    bounds = {'a':[-2, 3], 'b':[5, 6]}
    target_model = GPyRegression(parameter_names, bounds=bounds)
    x1 = np.random.uniform(*bounds['a'], n)
    x2 = np.random.uniform(*bounds['b'], n)
    x = np.column_stack((x1, x2))
    y = np.random.rand(n)
    target_model.update(x, y)

    # check acquisition without noise
    acq_noise_var = 0
    t = 1
    acquisition_method = acquisition.LCBSC(target_model, noise_var=acq_noise_var)
    new = acquisition_method.acquire(n2, t=t)
    assert np.allclose(new[1:, 0], new[0, 0])
    assert np.allclose(new[1:, 1], new[0, 1])

    # check acquisition with scalar noise
    acq_noise_var = 2
    t = 1
    acquisition_method = acquisition.LCBSC(target_model, noise_var=acq_noise_var)
    new = acquisition_method.acquire(n2, t=t)
    assert new.shape == (n2, n_params)
    assert np.all((new[:, 0] >= bounds['a'][0]) & (new[:, 0] <= bounds['a'][1]))
    assert np.all((new[:, 1] >= bounds['b'][0]) & (new[:, 1] <= bounds['b'][1]))

    # check acquisition with separate variance for dimensions
    acq_noise_var = np.random.uniform(0, 5, size=2)
    t = 1
    acquisition_method = acquisition.LCBSC(target_model, noise_var=acq_noise_var)
    new = acquisition_method.acquire(n2, t=t)
    assert new.shape == (n2, n_params)
    assert np.all((new[:, 0] >= bounds['a'][0]) & (new[:, 0] <= bounds['a'][1]))
    assert np.all((new[:, 1] >= bounds['b'][0]) & (new[:, 1] <= bounds['b'][1]))

    # check acquisition with arbitrary covariance matrix
    acq_noise_cov = np.random.rand(n_params, n_params) * 0.5
    acq_noise_cov += acq_noise_cov.T
    acq_noise_cov += n_params * np.eye(n_params)
    t = 1
    with pytest.raises(ValueError):
        acquisition.LCBSC(target_model, noise_var=acq_noise_cov)

    # test Uniform Acquisition
    t = 1
    acquisition_method = acquisition.UniformAcquisition(target_model,
                                                        noise_var=acq_noise_var)
    new = acquisition_method.acquire(n2, t=t)
    assert new.shape == (n2, n_params)
    assert np.all((new[:, 0] >= bounds['a'][0]) & (new[:, 0] <= bounds['a'][1]))
    assert np.all((new[:, 1] >= bounds['b'][0]) & (new[:, 1] <= bounds['b'][1]))
