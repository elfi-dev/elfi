import pytest
import elfi

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