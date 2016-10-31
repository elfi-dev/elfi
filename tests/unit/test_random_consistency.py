import numpy as np
from elfi import *
from elfi.examples.ma2 import *

from functools import partial
import pytest


def test_sub_streams():
    master_seed = 123
    sub_index = 0
    rands = []
    for i in range(2):
        state = get_substream_state(master_seed, sub_index)
        stream = np.random.RandomState(0)
        stream.set_state(state)
        rands.append(stream.randint(1000))
    assert rands[0] == rands[1]