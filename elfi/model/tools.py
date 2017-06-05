import subprocess
from functools import partial

import numpy as np

from elfi.utils import get_sub_seed


__all__ = ['vectorize', 'external_operation']


def run_vectorized(operation, *inputs, constants=None, batch_size=None, **kwargs):
    """Runs the operation as if it was vectorized over the individual runs in the batch.

    Helper for cases when you have an operation that does not support vector arguments.

    Parameters
    ----------
    operation : callable
        Operation that will be run `batch_size` times.
    inputs
        Inputs from the parent nodes.
    constants : tuple or int, optional
        A mask for constants in inputs, e.g. (0, 2) would indicate that the first and
        third input are constants. The constants will be passed as they are to each 
        operation call. 
    batch_size : int, optional
    kwargs

    Returns
    -------
    operation_output
        If batch_size > 1, a numpy array of outputs is returned
    """

    uses_batch_size = False if batch_size is None else True

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
    for index_in_batch in range(batch_size):
        # Prepare inputs for this run
        inputs_i = []
        for i_inpt, inpt in enumerate(inputs):
            if i_inpt in constants:
                inputs_i.append(inpt)
            else:
                inputs_i.append(inpt[index_in_batch])

        # Replace the batch_size with index_in_batch
        if uses_batch_size:
            kwargs['index_in_batch'] = index_in_batch

        runs.append(operation(*inputs_i, **kwargs))

    if batch_size == 1:
        return runs[0]
    else:
        return np.array(runs)


def vectorize(operation=None, constants=None):
    """Vectorizes an operation.

    Helper for cases when you have an operation that does not support vector arguments.

    Parameters
    ----------
    operation : callable, optional
        Operation to vectorize. Only pass this argument if you call this function 
        directly.
    constants : tuple, optional
        indexes of constants positional inputs for the operation. You can pass this as an
        argument for the decorator.

    Notes
    -----
    The decorator form does not always produce a pickleable object. The parallel execution
    requires the simulator to be pickleable. Therefore it is not recommended to use
    the decorator syntax unless you are using `dill` or a similar package.

    This is a convenience method and uses a for loop for vectorization. For best
    performance, one should aim to implement vectorized operations (by using e.g. numpy
    functions that are mostly vectorized) if at all possible.

    If the output from the operation is not a numpy array or if the shape of the output
    in different runs differs, the `dtype` of the returned numpy array will be `object`.

    If the node has a parameter `batch_index`, then also `run_index` will be added
    to the passed parameters that tells the current index of this run within the batch,
    i.e. 0 <= `run_index` < `batch_size`.

    Examples
    --------
    ::

        # Call directly (recommended)
        vectorized_simulator = elfi.tools.vectorize(simulator)

        # As a decorator without arguments
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

    """
    # Cases direct call or a decorator without arguments
    if callable(operation):
        return partial(run_vectorized, operation, constants=constants)

    # Decorator with parameters
    elif isinstance(operation, int):
        constants = tuple([operation])
    elif isinstance(operation, (tuple, list)):
        constants = tuple(operation)
    elif isinstance(constants, int):
        constants = tuple([constants])

    return partial(partial, run_vectorized, constants=constants)


def unpack_meta(*inputs, **kwinputs):
    if 'meta' in kwinputs:
        new_kwinputs = kwinputs['meta'].copy()
        new_kwinputs.update(kwinputs)
        kwinputs = new_kwinputs

    return inputs, kwinputs


def prepare_seed(*inputs, **kwinputs):
    if 'random_state' in kwinputs:
        # Get the seed for this batch, assuming np.RandomState instance
        seed = kwinputs['random_state'].get_state()[1][0]

        # Since we may not be the first operation to use this seed, lets generate a
        # a sub seed using this seed
        sub_seed_index = kwinputs.get('index_in_batch') or 0
        kwinputs['seed'] = get_sub_seed(np.random.RandomState(seed), sub_seed_index)

    return inputs, kwinputs


def stdout_to_array(stdout, *inputs, sep=' ', dtype=None, **kwinputs):
    """Converts a single row from stdout to np.array"""
    return np.fromstring(stdout, dtype=dtype, sep=sep)


def run_external(command, *inputs, process_result=None, prepare_inputs=None,
                 stdout=True, subprocess_kwargs=None, **kwinputs):
    """Run an external commmand (e.g. shell script, or executable) on a subprocess.

    See external_operation below for parameter descriptions.

    Returns
    -------
    output
    """

    inputs, kwinputs = unpack_meta(*inputs, **kwinputs)
    inputs, kwinputs = prepare_seed(*inputs, **kwinputs)
    if prepare_inputs:
        inputs, kwinputs = prepare_inputs(*inputs, **kwinputs)

    # Add arguments to the command
    try:
        command = command.format(*inputs, **kwinputs)
    except KeyError as e:
        raise KeyError('The requested keyword {} was not passed to the external '
                       'operation: "{}".'.format(str(e), command))

    subprocess_kwargs_ = dict(shell=True, check=True)
    subprocess_kwargs_.update(subprocess_kwargs or {})

    # Execute
    completed_process = subprocess.run(command, **subprocess_kwargs_)

    if stdout:
        completed_process = completed_process.stdout

    output = process_result(completed_process, *inputs, **kwinputs)

    return output


def external_operation(command, process_result=None, prepare_inputs=None, sep=' ',
                       stdout=True, subprocess_kwargs=None):
    """Wrap an external command as a Python callable (function).
    
    The external command can be e.g. a shell script, or an executable file.
    
    Parameters
    ----------
    command : str
        Command to execute. Arguments can be passed to the executable by using Python's
        format strings, e.g. `"myscript.sh {0} {batch_size} --seed {seed}"`. The command
        is expected to write to stdout. Since `random_state` is python specific object, a
        `seed` keyword argument will be available to operations that use `random_state`.
    process_result : callable, np.dtype, str, optional
        Callable result handler with a signature
        `output = callable(result, *inputs, **kwinputs)`. Here the `result` is either the
        stdout or `subprocess.CompletedProcess` depending on the stdout flag below. The
        inputs and kwinputs will come from ELFI. The default handler converts the stdout
        to numpy array with `array = np.fromstring(stdout, sep=sep)`. If `process_result`
        is `np.dtype` or a string, then the stdout data is casted to that type with
        `stdout = np.fromstring(stdout, sep=sep, dtype=process_result)`.
    prepare_inputs : callable, optional
        Callable with a signature `inputs, kwinputs = callable(*inputs, **kwinputs)`. The
        inputs will come from elfi.
    sep : str, optional
        Separator to use with the default `process_result` handler. Default is a space
        `' '`. If you specify your own callable to `process_result` this value has no
        effect.
    stdout : bool, optional
        Pass the `process_result` handler the stdout instead of the
        `subprocess.CompletedProcess` instance. Default is true.
    subprocess_kwargs : dict, optional
        Options for Python's `subprocess.run` that is used to run the external command.
        Defaults are `shell=True, check=True`. See the `subprocess` documentation for more
        details.
    
    Examples
    --------

    >>> import elfi
    >>> op = elfi.tools.external_operation('echo 1 {0}', process_result='int8')
    >>>
    >>> constant = elfi.Constant(123)
    >>> simulator = elfi.Simulator(op, constant)
    >>> simulator.generate()
    array([  1, 123], dtype=int8)
    
    Returns
    -------
    operation : callable
        ELFI compatible operation that can be used e.g. as a simulator.
    """

    if process_result is None or isinstance(process_result, (str, np.dtype)):
        fromstring_kwargs = dict(sep=sep)
        if isinstance(process_result, (str, np.dtype)):
            fromstring_kwargs['dtype'] = str(process_result)
        process_result = partial(stdout_to_array, **fromstring_kwargs)
        stdout = True

    if stdout is True:
        # Request stdout
        subprocess_kwargs = subprocess_kwargs or {}
        subprocess_kwargs['stdout'] = subprocess.PIPE

    return partial(run_external, command, process_result=process_result,
                   prepare_inputs=prepare_inputs, stdout=stdout,
                   subprocess_kwargs=subprocess_kwargs)
