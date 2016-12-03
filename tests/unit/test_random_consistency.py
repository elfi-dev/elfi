import numpy as np
from elfi import *
from examples.ma2 import *

from functools import partial
import pytest


def test_sub_streams():
    """Simple test that `get_substream_state` is pure
    """
    master_seed = 123
    sub_index = 0
    stream = np.random.RandomState(0)
    rands = []
    for i in range(2):
        state = get_substream_state(master_seed, sub_index)
        stream.set_state(state)
        rands.append(stream.randint(int(1e6)))
    assert rands[0] == rands[1]