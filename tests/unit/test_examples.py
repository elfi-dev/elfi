"""Simple running tests for examples."""

import os

import pytest

import elfi
from elfi.examples import arch, bdm, bignk, gauss, gnk, lotka_volterra, \
                          ricker, daycare, lorenz, ar1, mg1, toad, \
                          scratch_assay, stochastic_volatility_model


def test_bdm():
    """Currently only works in unix-like systems and with a cloned repository."""
    cpp_path = bdm.get_sources_path()

    do_cleanup = False
    if not os.path.isfile(cpp_path + '/bdm'):
        os.system('make -C {}'.format(cpp_path))
        do_cleanup = True

    assert os.path.isfile(cpp_path + '/bdm')

    # Remove the executable if it already exists
    if os.path.isfile('bdm'):
        os.system('rm bdm')

    with pytest.warns(RuntimeWarning):
        m = bdm.get_model()

    # Copy the file here to run the test
    os.system('cp {}/bdm .'.format(cpp_path))

    # Should no longer warn
    m = bdm.get_model()

    # Test that you can run the inference

    rej = elfi.Rejection(m, 'd', batch_size=100)
    rej.sample(20)

    # TODO: test the correctness of the result

    os.system('rm ./bdm')
    if do_cleanup:
        os.system('rm {}/bdm'.format(cpp_path))


def test_gauss():
    m = gauss.get_model()
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)


def test_gauss_1d_mean():
    params_true = [4]
    cov_matrix = [1]

    m = gauss.get_model(true_params=params_true, nd_mean=True, cov_matrix=cov_matrix)
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)


def test_gauss_2d_mean():
    params_true = [4, 4]
    cov_matrix = [[1, .5], [.5, 1]]

    m = gauss.get_model(true_params=params_true, nd_mean=True, cov_matrix=cov_matrix)
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)


def test_Ricker():
    m = ricker.get_model()
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20, quantile=0.1)


def test_Lorenz():
    m = lorenz.get_model()
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20, quantile=0.1)


def test_gnk():
    m = gnk.get_model()
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)


def test_bignk(stats_summary=['ss_octile']):
    m = bignk.get_model()
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)


def test_Lotka_Volterra():
    m = lotka_volterra.get_model(time_end=0.05)
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(10, quantile=0.5)


def test_daycare():
    m = daycare.get_model(time_end=0.05)
    rej = elfi.Rejection(m['d'], batch_size=10)
    rej.sample(10, quantile=0.5)


def test_ar1():
    m = ar1.get_model()
    rej = elfi.Rejection(m['d'], batch_size=10)
    rej.sample(10, quantile=0.5)


def test_mg1():
    m = mg1.get_model()
    rej = elfi.Rejection(m['d'], batch_size=10)
    rej.sample(10, quantile=0.5)


def test_toad():
    m = toad.get_model()
    rej = elfi.Rejection(m['d'], batch_size=10)
    rej.sample(10, quantile=0.5)


def test_stochastic_volatility_model():
    m = stochastic_volatility_model.get_model()
    rej = elfi.Rejection(m['d'], batch_size=10)
    rej.sample(10, quantile=0.5)


def test_arch():
    m = arch.get_model()
    rej = elfi.Rejection(m['d'], batch_size=10)
    rej.sample(10, quantile=0.5)


def test_scratch_assay():
    m = scratch_assay.get_model()
    rej = elfi.Rejection(m['d'], batch_size=10)
    rej.sample(10, quantile=0.5)
