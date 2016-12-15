import itertools
from functools import partial

import numpy as np

from elfi.utils import *
from elfi.storage import ElfiStore, LocalDataStore, MemoryStore
from elfi.graph import Node
from elfi import env

# TODO: enforce this?
DEFAULT_DATATYPE = np.float32


def prepare_store(store):
    """Takes in user-originated specifier for 'store' and
    returns a corresponding ElfiStore derivative or raises
    a value error.

    Parameters
    ----------
    store : various
        None : means data is not stored.
        ElfiStore derivative : stores data according to specification.
        String identifiers :
        "cache" : Creates a MemoryStore()
        Sliceable object : is converted to LocalDataStore(obj)

        Examples: local numpy array, h5py instance.
        The size of the object must be at least (n_samples, )  data.shape
        The slicing must be consistent:
            obj[sl] = d must guarantee that obj[sl] == d
            For example, an empty list will not guarantee this, but a pre-allocated will.
        See also: LocalDataStore

    Returns
    -------
    `ElfiStore` instance or None is store is None
    """

    if store is None:
        return None
    if isinstance(store, ElfiStore):
        return store
    if type(store) == str:
        if store.lower() == "cache":
            return MemoryStore()
        raise ValueError("Unknown store identifier '{}'".format(store))
    return LocalDataStore(store)


class DelayedOutputCache:
    """Handles a continuous list of delayed outputs for a node.
    """
    def __init__(self, node_id, store=None):
        """

        Parameters
        ----------
        node_id : str
            id of the node (`node.id`)
        store : various
            See prepare_store
        """
        self._delayed_outputs = []
        self._stored_mask = []
        self._store = prepare_store(store)
        self._node_id = node_id

    def __len__(self):
        l = 0
        for o in self._delayed_outputs:
            l += slen(get_key_slice(o.key))
        return l

    def append(self, output):
        """Appends output to cache/store

        """
        if len(self) != get_key_slice(output.key).start:
            raise ValueError('Appending a non matching slice')

        self._delayed_outputs.append(output)
        self._stored_mask.append(False)
        if self._store:
            self._store.write(output, done_callback=self._set_stored)

    def reset(self, new_node_id):
        del self._delayed_outputs[:]
        del self._stored_mask[:]
        if self._store is not None:
            self._store.reset(self._node_id)
        self._node_id = new_node_id

    def __getitem__(self, sl):
        """
        Returns the delayed data in slice `sl`
        """
        sl = to_slice(sl)
        outputs = self._get_output_datalist(sl)

        # Return the data_slice
        if len(outputs) == 0:
            empty = np.zeros(shape=(0,0))
            output = delayed(empty)
        elif len(outputs) == 1:
            output = outputs[0]
        else:
            key = reset_key_slice(outputs[0].key, sl)
            output = delayed(np.vstack)(tuple(outputs), dask_key_name=key)
        return output

    def _get_output_datalist(self, sl):
        data_list = []
        for i, output in enumerate(self._delayed_outputs):
            output_sl = get_key_slice(output.key)
            intsect_sl = slice_intersect(output_sl, sl)
            if slen(intsect_sl) == 0:
                continue

            if self._stored_mask[i] == True:
                output_data = self._store.read_data(self._node_id, output_sl)
            else:
                output_data = take_delayed(output, 'data')

            if slen(intsect_sl) != slen(output_sl):
                # Take a subset of the data-slice
                intsect_key = reset_key_slice(output_data.key, intsect_sl)
                sub_sl = slice_intersect(intsect_sl, offset=output_sl.start)
                output_data = delayed(operator.getitem)(output_data, sub_sl, dask_key_name=intsect_key)
            data_list.append(output_data)
        return data_list

    def _set_stored(self, key, result):
        """Inform that the result is computed for the `key`.

        Allows self to start using the stored delayed object

        Parameters
        ----------
        key : key of the original output
        result : future or concrete result of the output (currently not used)
        """
        output = [i for i,o in enumerate(self._delayed_outputs) if o.key == key]
        if len(output) != 1:
            # TODO: this error doesn't actually currently propagate into the main thread
            # Also make a separate case for len > 1
            raise LookupError('Cannot find output with the given key')
        i = output[0]
        self._stored_mask[i] = True


def to_output_dict(input_dict, **kwargs):
    output_dict = input_dict.copy()
    for k, v in kwargs.items():
        output_dict[k] = v
    return output_dict


substreams = itertools.count()

# TODO: this could have similar logic as utils.atleast_2d
def normalize_data(data, n=1):
    """Translates user-originated data into format compatible with the core.

    Parameters
    ----------
    data : any object
        User-originated data.
    n : int
        Number of times to replicate data (vectorization).

    Returns
    -------
    ret : np.ndarray

    If type(data) is not list, tuple or numpy.ndarray:
        ret.shape == (n, 1), ret[i][0] == data for all i
    If type(data) is list or tuple:
        data is converted to atleast 1D numpy array, after which
    If data.ndim == 1:
        If len(data) == n:
            ret.shape == (n, 1), ret[i][0] == data[i] for all i
        If len(data) != n:
            ret.shape == (n, len(data), ret[i] == data for all i
    If data.ndim > 1:
        If len(data) == n:
            ret == data
        If len(data) != n:
            ret.shape == (n, ) + data.shape, ret[i] == data for all i

    Examples
    --------
    Plain data
    >>> normalize_data(1, n=1)
    array([[1]])
    >>> normalize_data(1, n=2)
    array([[1],
           [1]])

    1D data
    >>> normalize_data([1], n=1)
    array([[1]])
    >>> normalize_data([1], n=2)
    array([[1],
           [1]])
    >>> normalize_data([1, 2], n=1)
    array([[1, 2]])
    >>> normalize_data([1, 2], n=2)
    array([[1],
           [2]])

    2D data
    >>> normalize_data([[1]], n=1)
    array([[1]])
    >>> normalize_data([[1]], n=2)
    array([[[1]],
    <BLANKLINE>
           [[1]]])
    >>> normalize_data([[1], [2]], n=1)
    array([[[1],
            [2]]])
    >>> normalize_data([[1], [2]], n=2)
    array([[1],
           [2]])
    """

    if isinstance(data, str):
        # TODO: could Antti comment on this?
        # numpy array initialization works unintuitively with strings
        data = np.array([[data]], dtype=object)
    else:
        data = np.atleast_1d(data)

    if data.ndim == 1:
        if data.shape[0] == n:
            data = data[:, None]
        else:
            data = data[None, :]
            if n > 1:
                data = np.vstack((data, ) * n)
    else:
        if data.shape[0] != n:
            data = data[None, :]
            if n > 1:
                data = np.vstack((data, ) * n)
    return data


def normalize_data_dict(dict, n):
    if dict is None:
        return None
    normalized = {}
    for k, v in dict.items():
        normalized[k] = normalize_data(v, n)
    return normalized


class Transform(Node):
    def __init__(self, name, transform, *parents, inference_task=None, store=None):
        """Transforms take `input_dict` as an argument and turn it into `output_dict`

        The `input_dict` will have a key "data", that contains a tuple where each parent
        in `parents` is replaced by the parent data.

        Parameters
        ----------
        name : string
            Name of the node
        transform : node transform function
            `transform(input_dict)` returns `output_dict`
            `input_dict` and `output_dict` must contain a key `"data"`
        *parents : tuple or list
            Parents of the operation node
        store : `OutputStore` instance

        Implementation details
        ----------------------
        - The outputs are assumed to be continuous from _start_index to _next_index
        - Parents have at least as much generated data as their children
        - Children may have less data generated than their parents
        - Using `with_values` will fail if parent data has been already generated
        - output indexes and batch sizes will match among all nodes in the same component
        - any given batch_size will be overridden by a parent batch_size if data for
          the parent had already been generated

        Future todo items
        -----------------
        - Allow parent output batch to be split in parts for the children. E.g. if 1M
          parameters were generated in one batch, simulations could still be done in e.g.
          batches of 100
        - should mismatching batch_size rather throw an error than adjust to parent
          batch_size?
        - think how n, batch_size and with_values work together in acquire and generate
        - prevent using with_values or batch_size if not leveled up?


        """
        inference_task = inference_task or env.inference_task()
        super(Transform, self).__init__(name, *parents, graph=inference_task)
        self._transform = transform

        # The index where outputs for this node are available
        self._start_index = 0
        self._cursor = 0
        # Keeps track of the resets
        self._num_resets = 0
        self._delayed_outputs = DelayedOutputCache(self.id, store)

    def __getitem__(self, sl):
        sl = to_slice(sl)
        return self._delayed_outputs.take_slice(sl)

    def level_up(self):
        """Generates delayed data so that it's at the same level with it's parents"""
        raise NotImplementedError

    # TODO: by default, get the computed output
    def acquire(self, n, starting=0, batch_size=None):
        """Acquires values from the start or from starting index.
        Generates new ones if needed and updates the cursor.

        Parameters
        ----------
        n : int
            number of samples
        starting : int
        batch_size : int

        Returns
        -------
        np.ndarray
        """
        sl = slice(starting, starting + n)
        if self._cursor < sl.stop:
            self.generate(sl.stop - self._cursor, batch_size=batch_size)
        return self[sl]

    # TODO: by default, get the computed output
    def generate(self, n, batch_size=None, with_values=None):
        """Generate n new values from the node. If all of the n values are going to be the
        same value, it is allowed to return just one value (see e.g. Constant).

        Parameters
        ----------
        n : int
            number of samples
        batch_size : desired batch size
            the batch sizes will correspond to batch sizes of the parents if
            they have already generated data
        with_values : dict(node_name: np.array)

        Returns
        -------
        n new values or a 1 value if all n values are the same
        """

        stop = self._cursor + n

        while self._next_index <= stop:
            default_batch_size = stop - self._next_index if batch_size is None else \
                batch_size
            self.get_or_create_delayed_output(self._next_index,
                                              default_batch_size,
                                              with_values)

        return self[slice(self._cursor, stop)]

    @property
    def _next_index(self):
        return self._delayed_outputs.next_index

    # TODO: by default, get the computed output
    def create_output(self, batch_size, take=None, with_values=None):
        """Adds on top of outputs created in this node"""
        out = self.get_or_create_delayed_output(self._next_index,
                                                batch_size,
                                                with_values)
        return out

    # TODO: by default, get the computed output
    def get_output(self, index, take=None, delayed=False):
        out = self._delayed_outputs.take(index, take)
        #if take is not None:
        #    out = take_delayed(out, 'data')
        #if delayed is False:
        #    out = out.compute()
        return out

    # TODO: separate this to two different
    def get_or_create_delayed_output(self, index, default_batch_size=None, with_values=None):
        if self._start_index > index:
            raise ValueError("There is no data available before index {}".format(index))

        if self._delayed_outputs[index] is None:
            # Produce a new output
            new_input = self._create_input_dict(index, default_batch_size, with_values)
            new_output = self._create_delayed_output(index,
                                                     new_input,
                                                     with_values)
            self._delayed_outputs[index] = new_output

        elif with_values is not None and self.name in with_values:
            raise ValueError("Corresponding data for dependency {} "
                             "has already been generated".format(self.name))

        return self._delayed_outputs[index]

    def _create_input_dict(self, index, default_batch_size, with_values=None):
        batch_size = None
        input_data = []
        for p in self.parents:
            p_out = p.get_or_create_delayed_output(index,
                                                   default_batch_size=default_batch_size,
                                                   with_values=with_values)
            p_bs = get_key_batch_size(p_out.key)

            # Determine batch_size from parents
            if batch_size is None:
                batch_size = p_bs
            elif batch_size != p_bs:
                raise ValueError("Incompatible batch_size {} in parent {}".format(p_bs, p.name))

            input_data.append(p_out)

        if len(self.parents) == 0:
            batch_size = default_batch_size

        return {
            "data": input_data,
            "n": batch_size,
            "index": index,
        }

    def _create_delayed_output(self, index, input_dict, with_values=None):
        """

        Parameters
        ----------
        index : int
        input_dict : dict
        with_values : dict {'node_name': np.array}

        Returns
        -------
        out : dask.delayed object

        """
        batch_size = input_dict["n"]
        with_values = with_values or {}
        dask_key = make_key(self.id, index, batch_size)
        if self.name in with_values:
            values = with_values[self.name]
            if values.ndim < 2:
                raise ValueError("with_values dimension is too small: {}".format(values.ndim))
            if len(values) > 1 and len(values) != batch_size:
                raise ValueError("with_values length don't match with the batch_size")
            # Set the data to with_values
            output = to_output_dict(input_dict,
                                    data=with_values[self.name])
            return delayed(output, name=dask_key)
        else:
            dinput = delayed(input_dict, pure=True)
            return delayed(self._transform)(dinput,
                                            dask_key_name=dask_key)

    @property
    def id(self):
        return make_key_id(self.inference_task.name, self.name, self.version)

    @property
    def inference_task(self):
        return self.graph

    @property
    def transform(self):
        return self._transform

    def set_transform(self, transform):
        """Sets the transform of the node directly
        """
        self._transform = transform

    @property
    def version(self):
        """Version of the node (currently number of resets)"""
        return self._num_resets

    def redefine(self, transform, *parents):
        """Redefines the transform of the node and the parents. Resets the data.

        Parameters
        ----------
        transform : new transform
        parents : new parents

        Returns
        -------

        """
        self.remove_parents()
        self.add_parents(parents)
        self._transform = transform
        self.reset()

    def reset(self, propagate=True):
        """Resets the data of the node

        Resets the node to a state as if no data was generated from it.
        If propagate is True (default) also resets its descendants

        Parameters
        ----------
        propagate : bool

        """
        if propagate:
            for c in self.children:
                c.reset()
        self._cursor = 0
        self._num_resets += 1
        self._delayed_outputs.reset(self.id)

    def _convert_to_node(self, obj, name):
        return Constant(name, obj)


class Operation(Transform):
    """Operation transforms parent data to a new data vector of the same length as
    their parents data.

    The transform is defined by the operation and the class attribute `operation_wrapper`,
    which wraps the operation to a transform.

    This class is a super class for LFI specific operations. The operations are callables
    whose signature is defined and tailored to serve the specific purpose of the
    respective subclass. See e.g. `class Summary(Operation)`

    Class variables
    ---------------
    operation_transform : callable(input_dict, operation)
        Wraps operations to transforms

    """
    operation_transform = None

    def __init__(self, name, operation, *args, **kwargs):
        self._operation = self._prepare_operation(operation, **kwargs)
        transform = self._make_transform(self._operation, **kwargs)
        super(Operation, self).__init__(name, transform, *args, **kwargs)

    def _prepare_operation(self, operation, **kwargs):
        return operation

    def _make_transform(self, operation, **kwargs):
        if self.__class__.operation_transform is not None:
            operation = partial(self.__class__.operation_transform, operation=operation)
        return operation

    def redefine(self, operation, *parents, **kwargs):
        operation = self._prepare_operation(operation, **kwargs)
        self._operation = operation
        transform = self._make_transform(operation, **kwargs)
        super(Operation, self).redefine(transform, *parents)

    @property
    def operation(self):
        return self._operation


"""
Operation mixins add additional functionality to the Operation class.
They do not define the actual operation but may add add keyword arguments
for the constructor. They may also add keys to `input_dict` and `output_dict`.
"""


def get_substream_state(master_seed, substream_index):
    """Returns PRNG internal state for the sub stream

    Parameters
    ----------
    master_seed : uint32
    substream_index : uint

    Returns
    -------
    out : tuple
    Random state for the sub stream as defined by numpy

    See Also
    --------
    'numpy.random.RandomState.get_state' for the representation of MT19937 state
    """
    # Fixme: In the future, allow MRG32K3a from https://pypi.python.org/pypi/randomstate
    seeds = np.random.RandomState(master_seed)\
        .randint(np.iinfo(np.uint32).max, size=substream_index+1)
    return np.random.RandomState(seeds[substream_index]).get_state()


class RandomStateMixin(Operation):
    """Makes Operation node stochastic.
    """
    def __init__(self, *args, **kwargs):
        super(RandomStateMixin, self).__init__(*args, **kwargs)

    def _create_input_dict(self, sl, **kwargs):
        dct = super(RandomStateMixin, self)._create_input_dict(sl, **kwargs)
        dct["random_state"] = self._get_random_state()
        return dct

    def _get_random_state(self):
        it = self.inference_task
        return delayed(get_substream_state, pure=True)(it.seed, it.new_substream_index())


class ObservedMixin(Operation):
    """Adds observed data to the class.
    """

    def __init__(self, *args, observed=None, **kwargs):
        super(ObservedMixin, self).__init__(*args, **kwargs)
        if observed is None:
            self.observed = self._inherit_observed()
        else:
            self.observed = normalize_data(observed, 1)

    def _inherit_observed(self):
        if len(self.parents) < 1:
            raise ValueError("There are no parents to inherit from")
        for parent in self.parents:
            if not hasattr(parent, "observed"):
                raise ValueError("Parent {} has no observed value to inherit".format(parent))
        observed = tuple([p.observed for p in self.parents])
        observed = self._transform({"data": observed, "n": 1})["data"]
        return observed


"""
Operation nodes
"""


class Constant(ObservedMixin, Operation):
    """
    Constant. Holds a constant value and returns only that when asked to generate data.
    Observed value is set also to the same value.
    """
    def __init__(self, name, value):
        """

        Parameters
        ----------
        value : constant value returned from generate
        """
        if type(value) in (tuple, list, np.ndarray):
            self.value = normalize_data(value, len(value))
        else:
            self.value = normalize_data(value, 1)
        v = self.value.copy()
        super(Constant, self).__init__(name, lambda input_dict: {"data": v}, observed=v)


# TODO: combine with random_wrapper
def simulator_transform(input_dict, operation):
    """Wraps a simulator function to a transformation.

    Parameters
    ----------
    operation: callable(*parent_data, batch_size, random_state)
        parent_data : numpy array
        batch_size : number of simulations to perform
        random_state : RandomState object
    input_dict: dict
        ELFI input_dict for transformations

    Notes
    -----
    It is crucial to use the provided RandomState object for generating the random
    quantities when running the simulator. This ensures that results are reproducible and
    inference will be valid.

    If the simulator is implemented in another language, one should extract the state
    of the random_state and use it in generating the random numbers.

    """
    # set the random state
    random_state = np.random.RandomState(0)
    random_state.set_state(input_dict["random_state"])
    batch_size = input_dict["n"]
    data = operation(*input_dict["data"], batch_size=batch_size, random_state=random_state)
    if not isinstance(data, np.ndarray):
        raise ValueError("Simulation operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(data)))
    if data.shape[0] != batch_size or len(data.shape) < 2:
        raise ValueError("Simulation operation output format incorrect." +
                " Expected shape == ({}, ...).".format(batch_size) +
                " Received shape == {}.".format(data.shape))
    return to_output_dict(input_dict, data=data, random_state=random_state.get_state())


class Simulator(ObservedMixin, RandomStateMixin, Operation):
    """Simulator node

    Parameters
    ----------
    name: string
    simulator: callable
    """
    operation_transform = simulator_transform


# TODO: rename to standard_wrapper
def summary_transform(input_dict, operation):
    batch_size = input_dict["n"]
    data = operation(*input_dict["data"])

    if not isinstance(data, np.ndarray):
        raise ValueError("Summary operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(data)))
    if data.shape[0] != batch_size or len(data.shape) < 2:
        raise ValueError("Summary operation output format incorrect." +
                " Expected shape == ({}, ...).".format(batch_size) +
                " Received shape == {}.".format(data.shape))
    return to_output_dict(input_dict, data=data)


class Summary(ObservedMixin, Operation):
    operation_transform = summary_transform


def discrepancy_transform(input_dict, operation):
    batch_size = input_dict["n"]
    data = operation(input_dict["data"], input_dict["observed"])
    if not isinstance(data, np.ndarray):
        raise ValueError("Discrepancy operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(data)))
    if data.shape != (batch_size, 1):
        raise ValueError("Discrepancy operation output format incorrect." +
                " Expected shape == ({}, 1).".format(batch_size) +
                " Received shape == {}.".format(data.shape))
    return to_output_dict(input_dict, data=data)


class Discrepancy(Operation):
    """The operation input has a tuple of data and tuple of observed
    """
    operation_transform = discrepancy_transform

    def _create_input_dict(self, sl, **kwargs):
        dct = super(Discrepancy, self)._create_input_dict(sl, **kwargs)
        dct["observed"] = tuple([p.observed for p in self.parents])
        return dct


if __name__ == "__main__":
    import doctest
    doctest.testmod()
