import numpy as np

import elfi


class TestVectorization:
    """Test operation vectorization operations 'as_vectorized_*'
    """

    @staticmethod
    @elfi.tools.vectorize
    def mock_simulator1(*data, random_state=None):
        return np.array((1,))

    @staticmethod
    def mock_simulator2(*data, batch_size=1, random_state=None):
        return np.ones((batch_size, 1))

    @staticmethod
    @elfi.tools.vectorize
    def mock_summary1(*data):
        return np.ones((1,))

    @staticmethod
    def mock_summary2(*data):
        n_sim = data[0].shape[0]
        return np.ones((n_sim, 1))

    @staticmethod
    @elfi.tools.vectorize
    def mock_discrepancy1(x, y):
        return np.ones((1,))

    @staticmethod
    def mock_discrepancy2(x, y):
        n_sim = x[0].shape[0]
        return np.ones((n_sim, 1))

    def test_annotated_simulator(self):
        input_data = [np.atleast_2d([[1], [2]]), np.atleast_2d([[3], [4]])]
        output1 = self.mock_simulator1(*input_data, batch_size=2)
        output2 = self.mock_simulator2(*input_data, batch_size=2)
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