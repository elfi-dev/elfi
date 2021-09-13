import logging
import os
import time

import numpy as np
import pytest
import scipy.stats as ss

import elfi
import elfi.clients.dask as dask
import elfi.clients.ipyparallel as eipp
import elfi.clients.multiprocessing as mp
import elfi.clients.native as native
import elfi.examples.gauss
import elfi.examples.ma2
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.acquisition import ExpIntVar, MaxVar, RandMaxVar
from elfi.model.extensions import ModelPrior

elfi.clients.native.set_as_default()


# Add command line options
def pytest_addoption(parser):
    parser.addoption(
        "--client",
        action="store",
        default="all",
        help="perform the tests for the specified client (default all)")


"""Functional fixtures"""


@pytest.fixture(scope="session", params=[eipp, dask, mp, native])
def client(request):
    """Provides a fixture for all the different supported clients
    """

    client_module = request.param
    client_name = client_module.__name__.split('.')[-1]
    use_client = request.config.getoption('--client')

    if use_client != 'all' and use_client != client_name:
        pytest.skip("Skipping client {}".format(client_name))

    try:
        client = client_module.Client()
    except BaseException:
        pytest.skip("Client {} not available".format(client_name))

    yield client

    # Run cleanup code here if needed


@pytest.fixture()
def with_all_clients(client):
    pre = elfi.get_client()
    elfi.client.set_client(client)

    yield

    elfi.client.set_client(pre)


@pytest.fixture()
def use_logging():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('elfi.executor').setLevel(logging.WARNING)
    logging.getLogger('elfi.compiler').setLevel(logging.WARNING)


@pytest.fixture()
def skip_travis():
    if "TRAVIS" in os.environ and os.environ['TRAVIS'] == "true":
        pytest.skip("Skipping this test in Travis CI due to very slow run-time. Tested "
                    "locally!")


"""Model fixtures"""


@pytest.fixture()
def simple_model():
    m = elfi.ElfiModel()
    elfi.Constant(10, model=m, name='tau')
    elfi.Prior('uniform', 0, m['tau'], size=1, model=m, name='k1')
    elfi.Prior('normal', m['k1'], size=3, model=m, name='k2')
    return m


@pytest.fixture()
def ma2():
    return elfi.examples.ma2.get_model()


@pytest.fixture()
def acq_maxvar():
    """Initialise a MaxVar fixture.

    Returns
    -------
    MaxVar
        Acquisition method.

    """
    gp, prior = _get_dependencies_acq_fn()

    # Initialising the acquisition method.
    method_acq = MaxVar(model=gp, prior=prior)
    return method_acq


@pytest.fixture()
def acq_randmaxvar():
    """Initialise a RandMaxVar fixture.

    Returns
    -------
    RandMaxVar
        Acquisition method.

    """
    gp, prior = _get_dependencies_acq_fn()

    # Initialising the acquisition method.
    method_acq = RandMaxVar(model=gp, prior=prior)
    return method_acq


@pytest.fixture()
def acq_expintvar():
    """Initialise an ExpIntVar fixture.

    Returns
    -------
    ExpIntVar
        Acquisition method.

    """
    gp, prior = _get_dependencies_acq_fn()

    # Initialising the acquisition method.
    method_acq = ExpIntVar(model=gp, prior=prior)
    return method_acq


def _get_dependencies_acq_fn():
    """Provide the requirements for the MaxVar-based acquisition function initialisation.

    Returns
    -------
    (GPy.model.GPRegression, elfi.model.extensions.ModelPrior)
        Tuple containing a fit gp and a prior.

    """
    mean = [4, 4]
    cov_matrix = [[1, .5], [.5, 1]]
    names_param = ['mu_0', 'mu_1']
    eps_prior = 5  # The prior's range indicator used in the Gaussian noise model.
    bounds_param = {'mu_0': (mean[0] - eps_prior, mean[0] + eps_prior),
                    'mu_1': (mean[1] - eps_prior, mean[1] + eps_prior)}

    # Initialising the prior.
    gm_2d = elfi.examples.gauss.get_model(true_params=mean, nd_mean=True, cov_matrix=cov_matrix)
    prior = ModelPrior(gm_2d)

    # Generating the coordinates and the values of the fitting data.
    n_pts_fit = 10
    x1 = np.random.uniform(*bounds_param['mu_0'], n_pts_fit)
    x2 = np.random.uniform(*bounds_param['mu_1'], n_pts_fit)
    x = np.column_stack((x1, x2))
    y = np.random.rand(n_pts_fit)

    # Fitting the gp with the generated points.
    gp = GPyRegression(names_param, bounds=bounds_param)
    gp.update(x, y)

    return gp, prior


def sleeper(sec, batch_size, random_state):
    secs = np.zeros(batch_size)
    for i, s in enumerate(sec):
        st = time.time()
        time.sleep(float(s))
        secs[i] = time.time() - st
    return secs


@pytest.fixture()
def multivariate_model(request):
    ndim = request.param

    def fun(x, batch_size, random_state):
        return np.sum(x, keepdims=True, axis=1)

    m = elfi.ElfiModel()
    elfi.Prior(ss.multivariate_normal, [0]*ndim, model=m, name='t1')
    elfi.Simulator(fun, m['t1'], observed=np.array([[0]]), model=m, name='sim')
    elfi.Distance('euclidean', m['sim'], model=m, name='d')
    return m


def no_op(data):
    return data


@pytest.fixture()
def sleep_model(request):
    """The true param will be half of the given sleep time."""
    ub_sec = request.param or .5
    m = elfi.ElfiModel()
    elfi.Constant(ub_sec, model=m, name='ub')
    elfi.Prior('uniform', 0, m['ub'], model=m, name='sec')
    elfi.Simulator(sleeper, m['sec'], model=m, name='slept')
    elfi.Summary(no_op, m['slept'], model=m, name='summary')
    elfi.Distance('euclidean', m['summary'], model=m, name='d')

    m.observed['slept'] = ub_sec / 2
    return m


def rowsummer(x, batch_size, random_state):
    return np.sum(x, keepdims=True, axis=1)


@pytest.fixture()
def multivariate_model(request):
    ndim = request.param
    m = elfi.ElfiModel()
    elfi.Prior(ss.multivariate_normal, [0]*ndim, model=m, name='t1')
    elfi.Simulator(rowsummer, m['t1'], observed=np.array([[0]]), model=m, name='sim')
    elfi.Distance('euclidean', m['sim'], model=m, name='d')
    return m


"""Helper fixtures"""


@pytest.fixture()
def distribution_test():
    def test_non_rvs_attr(attr, distribution, rvs, *args, **kwargs):
        # Run some tests that ensure outputs are coherent (similar style as with e.g.
        # scipy distributions)

        rvs_none, rvs1, rvs2 = rvs
        attr_fn = getattr(distribution, attr)

        # Test pdf
        attr_none = attr_fn(rvs_none, *args, **kwargs)
        attr1 = attr_fn(rvs1, *args, **kwargs)
        attr2 = attr_fn(rvs2, *args, **kwargs)

        # With size=1 the length should be 1
        assert len(attr1) == 1
        assert len(attr2) == 2

        assert attr1.shape[1:] == attr2.shape[1:]

        assert attr_none.shape == attr1.shape[1:]

        # With size=None we should get data that is not wrapped to any extra dim
        return attr_none, attr1, attr2

    def run(distribution, *args, rvs=None, **kwargs):

        if rvs is None:
            # Run some tests that ensure outputs are similar to e.g. scipy distributions
            rvs_none = distribution.rvs(*args, size=None, **kwargs)
            rvs1 = distribution.rvs(*args, size=1, **kwargs)
            rvs2 = distribution.rvs(*args, size=2, **kwargs)
        else:
            rvs_none, rvs1, rvs2 = rvs

        # Test that if rvs_none should be a scalar but is wrapped
        assert rvs_none.squeeze().ndim == rvs_none.ndim

        # With size=1 the length should be 1
        assert len(rvs1) == 1
        assert len(rvs2) == 2

        assert rvs1.shape[1:] == rvs2.shape[1:]

        # With size=None we should get data that is not wrapped to any extra dim
        # (possibly a scalar)
        assert rvs_none.shape == rvs1.shape[1:]

        rvs = (rvs_none, rvs1, rvs2)

        # Test pdf
        pdf_none, pdf1, pdf2 = test_non_rvs_attr('pdf', distribution, rvs, *args, **kwargs)
        # Should be a scalar
        assert pdf_none.ndim == 0

        if hasattr(distribution, 'logpdf'):
            logpdf_none, logpdf1, logpdf2 = test_non_rvs_attr('logpdf', distribution, rvs, *args,
                                                              **kwargs)
            assert np.allclose(logpdf_none, np.log(pdf_none))
            assert np.allclose(logpdf1, np.log(pdf1))
            assert np.allclose(logpdf2, np.log(pdf2))

        if hasattr(distribution, 'gradient_logpdf'):
            glpdf_none, glpdf1, glpdf2 = test_non_rvs_attr('gradient_logpdf', distribution, rvs,
                                                           *args, **kwargs)

    return run
