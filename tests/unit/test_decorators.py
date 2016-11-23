import numpy as np

from elfi.decorators import as_vectorized_simulator
from elfi.decorators import as_vectorized_summary
from elfi.decorators import as_vectorized_discrepancy

class Test_vectorization():
    """Test operation vectorization operations 'as_vectorized_*'
    """

    @staticmethod
    @as_vectorized_simulator
    def mock_simulator1(*data, prng=None):
        return np.array((1,))

    @staticmethod
    def mock_simulator2(*data, n_sim=1, prng=None):
        return np.ones((n_sim, 1))

    @staticmethod
    @as_vectorized_summary
    def mock_summary1(*data):
        return np.ones((1,))

    @staticmethod
    def mock_summary2(*data):
        n_sim = data[0].shape[0]
        return np.ones((n_sim, 1))

    @staticmethod
    @as_vectorized_discrepancy
    def mock_discrepancy1(x, y):
        return np.ones((1,))

    @staticmethod
    def mock_discrepancy2(x, y):
        n_sim = x[0].shape[0]
        return np.ones((n_sim, 1))

    def test_annotated_simulator(self):
        input_data = [np.atleast_2d([[1], [2]]), np.atleast_2d([[3], [4]])]
        output1 = self.mock_simulator1(*input_data, n_sim=2)
        output2 = self.mock_simulator2(*input_data, n_sim=2)
        assert np.array_equal(output1, output2)

    def test_annotated_summary(self):
        input_data = [np.atleast_2d([[1], [2]]), np.atleast_2d([[3], [4]])]
        output1 = self.mock_summary1(*input_data)
        output2 = self.mock_summary2(*input_data)
        assert np.array_equal(output1, output2)

    def test_annotated_discrepancy(self):
        x = (np.atleast_2d([[1], [2]]), np.atleast_2d([[3], [4]]))
        y = (np.atleast_2d([[5]]), np.atleast_2d([[6]]))
        output1 = self.mock_discrepancy1(x, y)
        output2 = self.mock_discrepancy2(x, y)
        assert np.array_equal(output1, output2)

