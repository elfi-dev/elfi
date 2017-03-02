import pytest
import elfi

@pytest.fixture()
def simple_model():
    m = elfi.ElfiModel()
    tau = elfi.Constant('tau', 10, model=m)
    k1 = elfi.Prior('k1', 'uniform', 0, tau, size=1, model=m)
    k2 = elfi.Prior('k2', 'normal', k1, size=3, model=m)
    return m
