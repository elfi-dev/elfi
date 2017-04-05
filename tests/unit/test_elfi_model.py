import pytest

import numpy as np
import scipy.stats as ss

import elfi.model.elfi_model as em


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
