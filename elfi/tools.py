from functools import partial
import numpy as np


__all__ = ['vectorize']


def run_vectorized(operation, *inputs, constants=None, batch_size=None, **kwargs):
    """Runs the operation as if it was vectorized over the individual runs in the batch.

    Helper for cases when you have an operation that does not support vector arguments.

    Parameters
    ----------
    operation : callable
        operation that will be run `batch_size` times.
    inputs
    constants : tuple or int, optional
        indexes of arguments in inputs that are constants
    batch_size : int, optional
    kwargs

    Notes
    -----
    This is an experimental feature.

    This is a convenience method and uses a for loop for vectorization. For best
    performance, one should aim to implement vectorized operations (by using e.g. numpy
    functions that are mostly vectorized) if at all possible.

    If the output from the operation is not a numpy array or if the shape of the output
    in different runs differs, the `dtype` of the returned numpy array will be `object`.

    Returns
    -------
    operation_output
        If batch_size > 1, a numpy array of outputs is returned
    """

    constants = [] if constants is None else list(constants)

    # Check input and set constants and batch_size if needed
    for i, inpt in enumerate(inputs):
        if i in constants:
            continue

        try:
            l = len(inpt)
        except:
            constants.append(i)
            l = 1

        if l != 1:
            if batch_size is None:
                batch_size = l
            elif batch_size != l:
                raise ValueError('Batch size {} does not match with input {} length of '
                                 '{}. Please check `constants` for the vectorize '
                                 'decorator.')

    # If batch_size is still `None` set it to 1 as no inputs larger than it were found.
    if batch_size is None:
        batch_size = 1

    # Run the operation batch_size times
    runs = []
    for i_run in range(batch_size):
        # Prepare inputs for this run
        inputs_i = []
        for i_inpt, inpt in enumerate(inputs):
            if i_inpt in constants:
                inputs_i.append(inpt)
            else:
                inputs_i.append(inpt[i_run])

        runs.append(operation(*inputs_i, **kwargs))

    if batch_size == 1:
        return runs[0]
    else:
        return np.array(runs)


def vectorize(operation=None, constants=None):
    """Vectorizes an operation

    Parameters
    ----------
    operation : callable, optional
        Operation to vectorize.
    constants : tuple, optional
        indexes of constants positional inputs for the operation.

    Examples
    --------
    ```
    @elfi.tools.vectorize
    def simulator(a, b, random_state=None):
        # Simulator code
        pass

    @elfi.tools.vectorize(constants=1)
    def simulator(a, constant, random_state=None):
        # Simulator code
        pass

    @elfi.tools.vectorize(1)
    def simulator(a, constant, random_state=None):
        # Simulator code
        pass

    @elfi.tools.vectorize(constants=(0,2))
    def simulator(constant0, b, constant2, random_state=None):
        # Simulator code
        pass
    ```

    """
    # Test if used directly as a decorator
    if callable(operation):
        return partial(run_vectorized, operation, constants=constants)
    # Cases where constants is given as a positional argument
    elif isinstance(operation, int):
        constants = tuple([operation])
    elif isinstance(operation, (tuple, list)):
        constants = tuple(operation)
    elif isinstance(constants, int):
        constants = tuple([constants])

    return partial(partial, run_vectorized, constants=constants)
