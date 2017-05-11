from subprocess import check_output
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
        operation that will be run `batch_size` times.
    inputs
        inputs from the parent nodes from ElfiModel
    constants : tuple or int, optional
        a mask for constants in inputs, e.g. (0, 2) would indicate that the first and 
        third input are constants. The constants will be passed as they are to each 
        operation call. 
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

    If the node has a parameter `batch_index`, then also `run_index` will be added
    to the passed parameters that tells the current index of this run within the batch,
    i.e. 0 <= `run_index` < `batch_size`.

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

        if 'batch_index' in kwargs:
            kwargs['run_index'] = i_run

        runs.append(operation(*inputs_i, **kwargs))

    if batch_size == 1:
        return runs[0]
    else:
        return np.array(runs)


def vectorize(operation=None, constants=None):
    """Vectorizes an operation.

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
    If you need to pickle the vectorized simulator (for parallel execution) and don't have
    `dill` or a similar package available, you must use the direct form. See the first
    example below.

    Examples
    --------
    ```

    # Call directly
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
    ```

    """
    # Test if used as a decorator without arguments or as a function call
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


def run_external(command, *inputs, prepare_arguments=None, process_output=None,
                 universal_newlines=True, **kwargs):
    """Run an external commmand (e.g. shell script, or executable) on a subprocess.
    
    Parameters
    ----------
    command : str
        Command to execute. Arguments can be passed to the executable by using Python's 
        format strings, e.g. `"myscript.sh {0} --seed {seed}"` where {0} will be replaced 
        with `inputs[0]` and {seed} with `kwargs['seed']`.
    inputs
    prepare_arguments : callable, optional
        Callable with a signature `args, kwargs = callable(*args, **kwargs)`
    process_output : callable, optional
        Callable with a signature `result = callable(stdout, **kwargs)`
    universal_newlines : bool, optional
        True causes the stdout to be a string. Otherwise it will be bytes.
    kwargs

    Returns
    -------
    stdout
    """

    if prepare_arguments:
        inputs, kwargs = prepare_arguments(*inputs, **kwargs)

    # Add arguments to the command
    try:
        command = command.format(*inputs, **kwargs)
    except KeyError as e:
        raise KeyError('The requested keyword {} was not passed to the external '
                       'operation: "{}".'.format(str(e), command))
    command_args = command.split()

    # Execute
    stdout = check_output(command_args, universal_newlines=universal_newlines)

    if process_output:
        stdout = process_output(stdout)

    return stdout


def prepare_seed(*args, **kwargs):
    if 'random_state' in kwargs:
        # Get the seed for this batch, assuming np.RandomState instance
        seed = kwargs['random_state'].get_state()[1][0]

        # Since we may not be the first operation to use this seed, lets generate a
        # a sub seed using this seed
        sub_seed_index = kwargs.get('run_index') or 0
        kwargs['seed'] = get_sub_seed(np.random.RandomState(seed), sub_seed_index)

    return args, kwargs


def external_operation(command, prepare_arguments=None, process_output=None, sep=' '):
    """Wrap an external command to an ELFI compatible Python callable.
    
    The external command can be e.g. a shell script, or an executable file.
    
    Parameters
    ----------
    command : str
        Command to execute. Arguments can be passed to the executable by using Python's 
        format strings, e.g. `"myscript.sh {0} {batch_size} --seed {seed}"`. The command 
        is expected to write to stdout. Since `random_state` is python specific object, a 
        `seed` keyword argument will be available to operations that use `random_state`.
    prepare_arguments : callable, optional
        Callable with a signature `args, kwargs = callable(*args, **kwargs)`
    process_output : callable, np.dtype, str, optional
        Callable with a signature `stdout = callable(stdout, **kwags)`. Default is to
        convert the stdout to numpy array with `stdout = np.fromstring(stdout, sep=sep).
        If `process_output` is `np.dtype` or a string, then the stdout data is casted to
        that type with `stdout = np.fromstring(stdout, sep=sep, dtype=process_output)`.
    sep : str, optional
        Separator to use with the default `process_output`. Default is a space `' '`.
        If you specify your own callable to `process_output` this value has no effect.
    
    Examples
    --------

    >>> import elfi
    >>> op = elfi.tools.external_operation('echo 1, {0}', process_output='int8')
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

    if prepare_arguments is None:
        prepare_arguments = prepare_seed

    if process_output is None or isinstance(process_output, (str, np.dtype)):
        fromstring_kwargs = dict(sep=sep)
        if isinstance(process_output, (str, np.dtype)):
            fromstring_kwargs['dtype'] = str(process_output)
        process_output = partial(np.fromstring, **fromstring_kwargs)

    return partial(run_external, command, prepare_arguments=prepare_arguments,
                   process_output=process_output)
