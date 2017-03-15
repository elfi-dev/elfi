import pytest
import os

import numpy as np
import scipy.stats as ss

import elfi
import examples.ma2 as ma2
from elfi.store import FileStore


def test_add_read_batch():
    m = ma2.get_model()
    output = m.generate(3, outputs=['t1', 't2', 'd', 'MA2'])

    fs = FileStore(model_name='.test')
    fs.add_batch(output, 1)
    batch = fs.read_batch(1)

    for key in output:
        assert np.array_equal(batch[key], output[key])

    batch.close()

    fs.destroy()

    assert os.path.exists(fs.basepath)
    assert not os.path.exists(fs.path)
