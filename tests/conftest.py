import pytest
import elfi
import time

import numpy as np

import examples.ma2
import elfi.clients.ipyparallel as eipp
import elfi.clients.native as native

elfi.clients.native.set_as_default()


@pytest.fixture(scope="session",
                params=[native, eipp])
def client(request):
    """Provides a fixture for all the different supported clients
    """
    client = request.param.Client()

    yield client

    # Run cleanup code here if needed


@pytest.fixture()
def with_all_clients(client):
    pre = elfi.get_client()
    elfi.client.reset_default(client)

    yield

    elfi.client.reset_default(pre)


@pytest.fixture()
def simple_model():
    m = elfi.ElfiModel()
    tau = elfi.Constant('tau', 10, model=m)
    k1 = elfi.Prior('k1', 'uniform', 0, tau, size=1, model=m)
    k2 = elfi.Prior('k2', 'normal', k1, size=3, model=m)
    return m


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
    ub = elfi.Constant('ub', ub_sec, model=m)
    sec = elfi.Prior('sec', 'uniform', 0, ub, model=m)
    slept = elfi.Simulator('slept', sleeper, sec, model=m)
    d = elfi.Discrepancy('d',  examples.ma2.discrepancy, slept, model=m)
    m.observed['slept'] = ub_sec/2
    return m
