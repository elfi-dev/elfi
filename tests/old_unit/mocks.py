import numpy as np

"""Mock objects for testing purposes

MockSimulator : callable vectorized simulator
MockSequentialSimulator : callable sequential simulator
MockSummary : callable vectorized summary operation
MockSequentialSummary : callable sequential summary operation
MockDiscrepancy : callable vectorized discrepancy operation
MockSequentialDiscrepancy : callable sequential discrepancy operation
"""

class MockSimulator():

    def __init__(self, rets):
        self.n_calls = 0
        self.n_ret = 0
        self.rets = rets

    def __call__(self, *data, batch_size=1, random_state=None):
        self.n_calls += 1
        if random_state is not None:
            random_state.rand()
        ret = np.zeros((batch_size, ) + self.rets[0].shape)
        for i in range(batch_size):
            ret[i] = self.rets[self.n_ret]
            self.n_ret += 1
        return ret


class MockSequentialSimulator():

    def __init__(self, rets):
        self.n_calls = 0
        self.n_ret = 0
        self.rets = rets

    def __call__(self, *data, random_state=None):
        self.n_calls += 1
        if random_state is not None:
            random_state.rand()
        ret = self.rets[self.n_ret]
        self.n_ret += 1
        return ret


class MockSummary():

    def __init__(self, rets):
        self.n_calls = 0
        self.n_ret = 0
        self.rets = rets

    def __call__(self, *data):
        self.n_calls += 1
        n_sim = data[0].shape[0]
        ret = np.zeros((n_sim, ) + self.rets[0].shape)
        for i in range(n_sim):
            ret[i] = self.rets[self.n_ret]
            self.n_ret += 1
        return ret


class MockSequentialSummary():

    def __init__(self, rets):
        self.n_calls = 0
        self.n_ret = 0
        self.rets = rets

    def __call__(self, *data):
        self.n_calls += 1
        ret = self.rets[self.n_ret]
        self.n_ret += 1
        return ret


class MockDiscrepancy():

    def __init__(self, rets):
        self.n_calls = 0
        self.n_ret = 0
        self.rets = rets

    def __call__(self, x, y):
        self.n_calls += 1
        n_sim = x[0].shape[0]
        ret = np.zeros((n_sim, 1))
        for i in range(n_sim):
            ret[i] = self.rets[self.n_ret]
            self.n_ret += 1
        return ret


class MockSequentialDiscrepancy():

    def __init__(self, rets):
        self.n_calls = 0
        self.n_ret = 0
        self.rets = rets

    def __call__(self, x, y):
        self.n_calls += 1
        ret = self.rets[self.n_ret]
        self.n_ret += 1
        return ret


