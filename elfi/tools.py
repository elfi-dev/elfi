from functools import partial
import numpy as np

# TODO: make a cleaner implementation
def vectorized_operation(operation, *input_data, batch_size=None, **kwargs):
    # batch_size is `None` only in cases where it can be inferred from `input_data`
    if batch_size is None:
        batch_size = 1
        for i_data in input_data:
            len_i = len(i_data)
            if isinstance(i_data, tuple):
                len_i = max([len(ij_data) for ij_data in i_data])
            batch_size = max(len_i, batch_size)

    output_data = None
    # One needs to use batch_size here, because e.g. priors may have no input_data
    for i in range(batch_size):
        input_data_i = []
        for i_data in input_data:
            # Case where the input_data is tuples of ndarrays
            if isinstance(i_data, tuple):
                val = tuple([ij_data[i] if len(ij_data) == batch_size else ij_data for ij_data in i_data])
            else:
                val = i_data[i] if len(i_data) == batch_size else i_data
            input_data_i.append(val)
        d = operation(*input_data_i, **kwargs)
        if not isinstance(d, np.ndarray):
            raise ValueError("Operation output type incorrect." +
                             "Expected type np.ndarray, received type {}".format(type(d)))
        if output_data is None:
            output_data = np.empty((batch_size,) + d.shape, d.dtype)
        output_data[i] = d
    return output_data


def vectorize(operation):
    """Vectorizes an operation
    """
    return partial(vectorized_operation, operation)
