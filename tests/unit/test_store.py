import pytest
import os

import numpy as np
import scipy.stats as ss

import elfi
import examples.ma2 as ma2
from elfi.store import FileStore, NpyFileAppender


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


def test_npy_file_appender():
    filename = 'test.npy'

    original = np.random.rand(3, 2)

    f = NpyFileAppender(filename, 'w')
    f.append(original)
    f.close()
    loaded = np.load(filename)
    assert np.array_equal(original, loaded)

    # Test appending and reading
    f = NpyFileAppender(filename, 'a')
    append = np.random.rand(100, 2)
    f.append(append)
    f.flush()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, append], loaded)

    append2 = np.random.rand(23, 2)
    f.append(append2)
    f.close()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, append, append2], loaded)

    # Try that truncation works
    f = NpyFileAppender(filename, 'w')
    f.append(append)
    f.close()
    loaded = np.load(filename)
    assert np.array_equal(append, loaded)

    os.remove(filename)