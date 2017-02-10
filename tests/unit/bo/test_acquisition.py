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
        assert acq.finished is False

    def test_add_base(self):
        model = MockModel()
        acq1 = MockAcquisition(model, n_samples=1, val=np.array([3]))
        acq2 = MockAcquisition(model, n_samples=1, val=np.array([4]))
        sched = acq1 + acq2
        assert sched.samples_left == 2
        r1 = sched.acquire(1)
        assert r1[0] == 3
        assert sched.samples_left == 1
        assert sched.finished is False
        r2 = sched.acquire(1)
        assert r2[0] == 4
        assert sched.samples_left == 0
        assert sched.finished is True

    def test_add_sched(self):
        model = MockModel()
        acq1 = MockAcquisition(model, n_samples=1, val=np.array([3]))
        acq2 = MockAcquisition(model, n_samples=1, val=np.array([4]))
        acq3 = MockAcquisition(model, n_samples=1, val=np.array([5]))
        sched = acq1 + acq2
        assert sched.samples_left == 2
        sched += acq3
        assert sched.samples_left == 3
        r1 = sched.acquire(1)
        assert r1[0] == 3
        assert sched.samples_left == 2
        assert sched.finished is False
        r2 = sched.acquire(1)
        assert r2[0] == 4
        assert sched.samples_left == 1
        assert sched.finished is False
        r3 = sched.acquire(1)
        assert r3[0] == 5
        assert sched.samples_left == 0
        assert sched.finished is True

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

    def test_unreached_raises_error(self):
        model = MockModel()
        acq1 = MockAcquisition(model, n_samples=None, val=np.array([3]))
        acq2 = MockAcquisition(model, n_samples=1, val=np.array([4]))
        try:
            sched = acq1 + acq2
        except ValueError:
            return
        assert False

