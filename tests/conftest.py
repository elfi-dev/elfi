import pytest
import elfi

import elfi.clients.ipyparallel as eipp
import elfi.clients.native as native

elfi.clients.native.set_as_default()


@pytest.fixture(scope="session",
                params=[native, eipp])
def client(request):
    """Provdes a fixture for all the different supported clients
    """
    client = request.param.Client()
    yield client

    # Run cleanup code here if needed