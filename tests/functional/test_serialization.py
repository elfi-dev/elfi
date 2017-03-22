import pytest
import pickle

import numpy as np
import scipy.stats as ss

from elfi.client import ClientBase
from elfi.executor import Executor
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


def test_pickle_ma2_compiled_and_loaded(ma2):
    compiled = ClientBase.compile(ma2.source_net, ['d'])
    loaded = ClientBase.load_data(compiled, ma2.computation_context, 0)

    np.random.seed(0)
    result = Executor.execute(loaded)
    res1 = result['d']

    serialized = pickle.dumps(loaded)
    loaded = pickle.loads(serialized)

    np.random.seed(0)
    result = Executor.execute(loaded)
    res2 = result['d']

    assert np.array_equal(res1, res2)