from functools import partial
import pytest

import numpy as np

import elfi.v2.network as ev2

def test_me():
    prior = ev2.Prior('t2', 'uniform', 0, 3)
