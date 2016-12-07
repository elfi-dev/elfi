from functools import partial
import numpy as np


def vectorized_operation(operation, *input_data, batch_size=1, **kwargs):
    data = None
    for i in range(batch_size):
        input_data_i = []
        for i_data in input_data:
            # Case where the input_data is tuples of ndarrays
            if isinstance(i_data, tuple):
                val = tuple([elem[i] for elem in i_data])
            else:
                val = i_data[i]
            input_data_i.append(val)
        d = operation(*input_data_i, **kwargs)
        if not isinstance(d, np.ndarray):
            raise ValueError("Operation output type incorrect." +
                             "Expected type np.ndarray, received type {}".format(type(d)))
        if data is None:
            data = np.empty((batch_size,) + d.shape, d.dtype)
        data[i] = d
    return data


def vectorize(operation):
    """Vectorizes an operation
    """
    return partial(vectorized_operation, operation)
