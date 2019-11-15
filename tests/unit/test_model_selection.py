import pytest

import elfi
from elfi.examples import gauss, ma2


@pytest.mark.slowtest
def test_compare_models():
    m = gauss.get_model()
    res1 = elfi.Rejection(m['d']).sample(100)

    # use less informative prior
    m['mu'].become(elfi.Prior('uniform', -10, 50))
    res2 = elfi.Rejection(m['d']).sample(100)

    # use different simulator
    m['gauss'].become(elfi.Simulator(ma2.MA2, m['mu'], m['sigma'], observed=m.observed['gauss']))
    res3 = elfi.Rejection(m['d']).sample(100)

    p = elfi.compare_models([res1, res2, res3])
    assert p[0] > p[1]
    assert p[1] > p[2]
