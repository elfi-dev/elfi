import numpy as np
from functools import partial

from elfi.core import simulator_operation

class MockSimulator():

    def __init__(self, rets):
        self.n_calls = 0
        self.args = list()
        self.kwargs = list()
        self.rets = rets

    def __call__(self, *args, **kwargs):
        self.args.append(args)
        self.kwargs.append(kwargs)
        kwargs["prng"].rand()
        ret = self.rets[self.n_calls]
        self.n_calls += 1
        return ret


class Test_simulator_operation():

    def test_vectorized(self):
        ret1 = np.atleast_2d([[5], [6]])
        mock = MockSimulator([ret1])
        prng = np.random.RandomState(1234)
        input_dict = {
                "n": 2,
                "data": [np.atleast_2d([[1], [2]]),
                         np.atleast_2d([[3], [4]])],
                "random_state": prng.get_state()
                }
        output_dict = simulator_operation(mock, True, input_dict)
        prng.rand()
        print(output_dict)
        assert mock.n_calls == 1
        assert output_dict["n"] == 2
        assert np.array_equal(output_dict["data"], ret1)
        new_state = prng.get_state()
        assert output_dict["random_state"][0] == new_state[0]
        assert np.array_equal(output_dict["random_state"][1], new_state[1])
        assert output_dict["random_state"][2] == new_state[2]
        assert output_dict["random_state"][3] == new_state[3]
        assert output_dict["random_state"][4] == new_state[4]

    def test_sequential(self):
        ret1 = np.array([5])
        ret2 = np.array([6])
        mock = MockSimulator([ret1, ret2])
        prng = np.random.RandomState(1234)
        input_dict = {
                "n": 2,
                "data": [np.atleast_2d([[1], [2]]),
                         np.atleast_2d([[3], [4]])],
                "random_state": prng.get_state()
                }
        output_dict = simulator_operation(mock, False, input_dict)
        prng.rand()
        prng.rand()
        print(output_dict)
        assert mock.n_calls == 2
        assert output_dict["n"] == 2
        assert np.array_equal(output_dict["data"], np.vstack((ret1, ret2)))
        new_state = prng.get_state()
        assert output_dict["random_state"][0] == new_state[0]
        assert np.array_equal(output_dict["random_state"][1], new_state[1])
        assert output_dict["random_state"][2] == new_state[2]
        assert output_dict["random_state"][3] == new_state[3]
        assert output_dict["random_state"][4] == new_state[4]

