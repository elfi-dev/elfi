import pytest

import numpy as np
import scipy.stats as ss

import elfi.model.elfi_model as em
import examples.ma2 as ma2


@pytest.mark.usefixtures('with_all_clients')
def test_generate():
    n_gen = 10

    m = ma2.get_model()
    d = m.get_reference('d')
    res = d.generate(n_gen)

    assert res.shape[0] == n_gen
    assert res.ndim == 1


@pytest.mark.usefixtures('with_all_clients')
def test_observed():
    true_params = [.6, .2]
    m = ma2.get_model(100, true_params=true_params)
    y = m.observed['MA2']
    S1 = m.get_reference('S1')
    S2 = m.get_reference('S2')

    S1_observed = ma2.autocov(y)
    S2_observed = ma2.autocov(y, 2)

    assert np.array_equal(S1.observed, S1_observed)
    assert np.array_equal(S2.observed, S2_observed)


def test_node_reference_str():
    # This is important because it is used when passing NodeReferences as InferenceMethod
    # arguments
    ref = em.NodeReference(name='test')
    assert str(ref) == 'test'


def test_name_determination():
    node = em.NodeReference()
    assert node.name == 'node'

    # Works without spaces
    node2=em.NodeReference()
    assert node2.name == 'node2'

    # Does not give the same name
    node = em.NodeReference()
    assert node.name != 'node'

    # Works with sub classes
    pri = em.Prior()
    assert pri.name == 'pri'

    # Assigns random names when the name isn't self explanatory
    nodes = []
    for i in range(5):
        nodes.append(em.NodeReference())

    for i in range(1,5):
        assert nodes[i-1].name != nodes[i].name
