import numpy as np
import pytest

import elfi
import elfi.methods.bo.acquisition as acquisition
from elfi.methods.bo.gpy_regression import GPyRegression
import matplotlib.pyplot as plt


@pytest.mark.usefixtures('with_all_clients')
def test_BO(ma2):
    # Log transform of the distance usually smooths the distance surface
    log_d = elfi.Operation(np.log, ma2['d'], name='log_d')

    n_init = 20
    res_init = elfi.Rejection(log_d, batch_size=5).sample(n_init, quantile=1)

    bounds = {n: (-2, 2) for n in ma2.parameter_names}
    bo = elfi.BayesianOptimization(
        log_d, initial_evidence=res_init.outputs, update_interval=10, batch_size=5, bounds=bounds)
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

    assert np.array_equal(bo.target_model._gp.X[:n_init, 0], res_init.samples_array[:, 0])


@pytest.mark.usefixtures('with_all_clients')
def test_async(ma2):
    bounds = {n: (-2, 2) for n in ma2.parameter_names}
    bo = elfi.BayesianOptimization(
        ma2, 'd', initial_evidence=0, update_interval=2, batch_size=2, bounds=bounds, async=True)
    samples = 5
    bo.infer(samples)


@pytest.mark.usefixtures('with_all_clients')
def test_BO_works_with_zero_init_samples(ma2):
    log_d = elfi.Operation(np.log, ma2['d'], name='log_d')
    bounds = {n: (-2, 2) for n in ma2.parameter_names}
    bo = elfi.BayesianOptimization(
        log_d, initial_evidence=0, update_interval=4, batch_size=2, bounds=bounds)
    assert bo.target_model.n_evidence == 0
    assert bo.n_evidence == 0
    assert bo.n_precomputed_evidence == 0
    assert bo.n_initial_evidence == 0
    samples = 4
    bo.infer(samples)
    assert bo.target_model.n_evidence == samples
    assert bo.n_evidence == samples
    assert bo.n_precomputed_evidence == 0
    assert bo.n_initial_evidence == 0


def test_acquisition():
    n_params = 2
    n = 10
    n2 = 5
    parameter_names = ['a', 'b']
    bounds = {'a': [-2, 3], 'b': [5, 6]}
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
    acquisition_method = acquisition.UniformAcquisition(target_model, noise_var=acq_noise_var)
    new = acquisition_method.acquire(n2, t=t)
    assert new.shape == (n2, n_params)
    assert np.all((new[:, 0] >= bounds['a'][0]) & (new[:, 0] <= bounds['a'][1]))
    assert np.all((new[:, 1] >= bounds['b'][0]) & (new[:, 1] <= bounds['b'][1]))


class Test_MaxVar:
    """Using the acq_maxvar fixture.

    NOTES
    -----
    - The RandMaxVar acquisition is performed on a 2D Gaussian noise model.
    """

    def test_acq_bounds(self, acq_maxvar):
        n_pts_acq = 10
        bounds = acq_maxvar.model.bounds

        # Acquiring points
        x_acq = acq_maxvar.acquire(n_pts_acq)

        # Checking if the acquired points are within the bounds.
        assert np.all(x_acq >= bounds[0][0])
        assert np.all(x_acq <= bounds[0][1])

    def test_gradient(self, acq_maxvar):
        # Enabling/disabling the visualisation of the gradients.
        vis = False
        # Acquiring some points to initialise the acquisition method's params.
        n_pts_acq = 10
        acq_maxvar.acquire(n_pts_acq)

        # Partitioning the axes.
        bounds = acq_maxvar.model.bounds
        n_pts = 50
        lins_dim1, d_dim1 = np.linspace(*bounds[0], num=n_pts, retstep=True)
        lins_dim2, d_dim2 = np.linspace(*bounds[1], num=n_pts, retstep=True)

        # Computing the gradient using the acquisition method's class.
        evals = np.zeros(shape=(n_pts, n_pts))
        grads_maxvar = np.zeros(shape=(2, n_pts, n_pts))
        for idx_dim1, coord_dim1 in enumerate(lins_dim1):
            for idx_dim2, coord_dim2 in enumerate(lins_dim2):
                coord = coord_dim1, coord_dim2
                evals[idx_dim1, idx_dim2] = acq_maxvar.evaluate(coord)[0]
                grads_maxvar[:, idx_dim1, idx_dim2] \
                    = acq_maxvar.evaluate_gradient(coord)[0]

        # Computing the gradient via a finite difference method.
        grads_np = np.gradient(evals, d_dim1, d_dim2)
        grads_np = np.array(grads_np)

        if vis:
            # Plotting the computed gradients for the visual comparison.
            fig, arr_ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
            fig.tight_layout(pad=2.0)
            im_1 = arr_ax[0, 0].imshow(grads_maxvar[0], cmap='hot',
                interpolation='nearest')
            fig.colorbar(im_1, ax=arr_ax[0, 0])
            arr_ax[0, 0].set_title('Finite Difference, dim_1 gradient')
            im_2 = arr_ax[0, 1].imshow(grads_maxvar[1], cmap='hot',
                interpolation='nearest')
            fig.colorbar(im_2, ax=arr_ax[0, 1])
            arr_ax[0, 1].set_title('Finite Difference, dim_2 gradient')
            im_3 = arr_ax[1, 0].imshow(grads_np[0], cmap='hot',
                interpolation='nearest')
            fig.colorbar(im_3, ax=arr_ax[1, 0])
            arr_ax[1, 0].set_title('MaxVar, dim_1 gradient')
            im_4 = arr_ax[1, 1].imshow(grads_np[1], cmap='hot',
                interpolation='nearest')
            fig.colorbar(im_4, ax=arr_ax[1, 1])
            arr_ax[1, 1].set_title('MaxVar, dim_2 gradient')
            plt.show()

        # Calculating the absolute error.
        diff_dim_1 = np.sum(np.absolute(grads_np[0] - grads_maxvar[0]))
        diff_dim_2 = np.sum(np.absolute(grads_np[1] - grads_maxvar[1]))

        # Summing the norms of the gradient functions.
        # - Taking the average of the two results for the upcoming comparison.
        sum_grad_np_dim_1 = np.sum(np.absolute(grads_np[0]))
        sum_grad_np_dim_2 = np.sum(np.absolute(grads_np[1]))
        sum_grad_maxvar_dim_1 = np.sum(np.absolute(grads_np[0]))
        sum_grad_maxvar_dim_2 = np.sum(np.absolute(grads_np[1]))
        sum_grad_dim_1 = np.average([sum_grad_np_dim_1, sum_grad_maxvar_dim_1])
        sum_grad_dim_2 = np.average([sum_grad_np_dim_2, sum_grad_maxvar_dim_2])

        # Comparison the difference in the gradients w.r.t. the norm sums.
        leftover_dim_1 = diff_dim_1 / sum_grad_dim_1
        leftover_dim_2 = diff_dim_2 / sum_grad_dim_2

        # Specifying the threshold:
        # - The threshold value is the volume/mass comparison;
        #   i.e., the volume of the gradients' difference is compared to
        #   the volume of the gradient function.
        # - The test passes if the gradients are similar;
        #   i.e., the ratio is small.
        threshold_grad_diff = 0.25
        assert leftover_dim_1 < threshold_grad_diff
        assert leftover_dim_2 < threshold_grad_diff


class Test_RandMaxVar:
    """Using the acq_randmaxvar fixture.

    NOTES
    -----
    - The RandMaxVar acquisition is performed on a 2D Gaussian noise model.
    """

    def test_acq_bounds(self, acq_randmaxvar):
        n_pts_acq = 10
        bounds = acq_randmaxvar.model.bounds

        # Acquiring points
        x_acq = acq_randmaxvar.acquire(n_pts_acq)

        # Checking if the acquired points are within the bounds.
        assert np.all(x_acq >= bounds[0][0])
        assert np.all(x_acq <= bounds[0][1])
