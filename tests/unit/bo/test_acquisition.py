import sys
import numpy as np
from elfi.bo.acquisition import *

class MockModel():
    input_dim = 1
    bounds = ((0, 1),)

    def evaluate(self, x):
        return 0.0


class MockAcquisition(AcquisitionBase):

    def __init__(self, *args, val=1, **kwargs):
        self.val = val
        super(MockAcquisition, self).__init__(*args, **kwargs)

    def acquire(self, n_values, pending_locations=None):
        ret = super(MockAcquisition, self).acquire(n_values, pending_locations)
        for i in range(self.n_values):
            ret[i] = self.val
        return ret


class Test_acquisition_base_and_schedule():

    def test_init(self):
        model = MockModel()
        acq = AcquisitionBase(model)
        assert acq.finished is False
        assert acq.n_acquired == 0
        s1 = acq.samples_left
        loc = acq.acquire(2)
        assert loc.shape == (2, 1)
        s2 = acq.samples_left
        assert s1 == s2
        assert acq.n_acquired == 2

    def test_add(self):
        model = MockModel()
        acq1 = MockAcquisition(model, n_samples=1, val=np.array([3]))
        acq2 = MockAcquisition(model, n_samples=1, val=np.array([4]))
        sched = acq1 + acq2
        r1 = sched.acquire(1)
        assert r1[0] == 3
        r2 = sched.acquire(1)
        assert r2[0] == 4

    def test_reaching_end_raises_error(self):
        model = MockModel()
        acq1 = MockAcquisition(model, n_samples=1, val=np.array([3]))
        acq2 = MockAcquisition(model, n_samples=1, val=np.array([4]))
        sched = acq1 + acq2
        r = sched.acquire(1)
        r = sched.acquire(1)
        try:
            r = sched.acquire(1)
        except IndexError:
            return
        assert False

    def test_different_models_raises_error(self):
        model1 = MockModel()
        model2 = MockModel()
        acq1 = MockAcquisition(model1, n_samples=1, val=np.array([3]))
        acq2 = MockAcquisition(model2, n_samples=1, val=np.array([4]))
        try:
            sched = acq1 + acq2
        except ValueError:
            return
        assert False

    def test_unreached_raises_error(self):
        model = MockModel()
        acq1 = MockAcquisition(model, n_samples=None, val=np.array([3]))
        acq2 = MockAcquisition(model, n_samples=1, val=np.array([4]))
        try:
            sched = acq1 + acq2
        except ValueError:
            return
        assert False

