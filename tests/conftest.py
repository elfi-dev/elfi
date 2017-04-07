import pytest
import elfi
import time
import logging

import numpy as np

import examples.ma2
import elfi.clients.ipyparallel as eipp
import elfi.clients.native as native

elfi.clients.native.set_as_default()


# Add command line options
def pytest_addoption(parser):
    parser.addoption("--client", action="store", default="all",
        help="perform the tests for the specified client (default all)")

    parser.addoption("--skipslow", action="store_true",
        help="skip slow tests")


@pytest.fixture(scope="session",
                params=[native, eipp])
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
    except:
        pytest.skip("Client {} not available".format(client_name))

    yield client

    # Run cleanup code here if needed


@pytest.fixture()
def with_all_clients(client):
    pre = elfi.get_client()
    elfi.client.reset_default(client)

    yield

    elfi.client.reset_default(pre)


@pytest.fixture()
def use_logging():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('elfi.executor').setLevel(logging.WARNING)


# Model fixtures


@pytest.fixture()
def simple_model():
    m = elfi.ElfiModel()
    elfi.Constant(10, model=m, name='tau')
    elfi.Prior('uniform', 0, m['tau'], size=1, model=m, name='k1')
    elfi.Prior('normal', m['k1'], size=3, model=m, name='k2')
    return m


@pytest.fixture()
def ma2():
    return examples.ma2.get_model()


def sleeper(sec, batch_size, random_state):
    for s in sec:
        time.sleep(float(s))
    return sec


@pytest.fixture()
def sleep_model(request):
    """The true param will be half of the given sleep time

    """
    ub_sec = request.param or .5
    m = elfi.ElfiModel()
    elfi.Constant(ub_sec, model=m, name='ub')
    elfi.Prior('uniform', 0, m['ub'], model=m, name='sec')
    elfi.Simulator(sleeper, m['sec'], model=m, name='slept')
    elfi.Discrepancy(examples.ma2.discrepancy, m['slept'], model=m, name='d')

    m.observed['slept'] = ub_sec/2
    return m