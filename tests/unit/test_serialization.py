import pytest
import pickle

import numpy as np
import scipy.stats as ss

import elfi
from elfi.native_client import Client
import examples.ma2 as ma2


def test_pickle_ma2():
    m = ma2.get_model()
    d = m.get_reference('d')

    np.random.seed(0)
    res1 = d.generate(10)

    serialized = pickle.dumps(m)
    m = pickle.loads(serialized)
    d = m.get_reference('d')

    np.random.seed(0)
    res2 = d.generate(10)

    assert np.array_equal(res1, res2)


def test_pickle_ma2_compiled_and_loaded():
    m = ma2.get_model()
    compiled = Client.compile(m, 'd')
    loaded = Client.load_data(m.computation_context, compiled, (0, 10))

    np.random.seed(0)
    res_dict = Client.execute(loaded)
    res1 = res_dict['d']['output']

    serialized = pickle.dumps(loaded)
    loaded = pickle.loads(serialized)

    np.random.seed(0)
    res_dict = Client.execute(loaded)
    res2 = res_dict['d']['output']

    assert np.array_equal(res1, res2)