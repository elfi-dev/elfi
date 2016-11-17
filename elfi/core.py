import numpy as np
import uuid

import operator
from tornado import gen

from dask.delayed import delayed
import itertools
from functools import partial
from collections import defaultdict
from elfi.utils import to_slice, slice_intersect, slen
from elfi import env


DEFAULT_DATATYPE = np.float32


class Node(object):
    """A base class representing Nodes in a graphical model.
    This class is inherited by all types of nodes in the model.

    Attributes
    ----------
    name : string
    parents : list of Nodes
    children : list of Nodes
    """
    def __init__(self, name, *parents):
        self.name = name
        self.parents = []
        self.children = []
        self.add_parents(parents)

    def reset(self, *args, **kwargs):
        pass

    def add_parents(self, nodes):
        for n in self.node_list(nodes):
            self.add_parent(n)

    def add_parent(self, node, index=None, index_child=None):
        """Adds a parent and assigns itself as a child of node. Only add if new.

        Parameters
        ----------
        node : Node or None
            If None, this function will not do anything
        index : int
            Index in self.parents where to insert the new parent.
        index_child : int
            Index in self.children where to insert the new child.
        """
        if node is None:
            return
        node = self._ensure_node(node)
        if node in self.descendants:
            raise ValueError("Cannot have cyclic graph structure.")
        if not node in self.parents:
            if index is None:
                index = len(self.parents)
            else:
                if index < 0 or index > len(self.parents):
                    raise ValueError("Index out of bounds.")
            self.parents.insert(index, node)
        node._add_child(self, index_child)

    def _add_child(self, node, index=None):
        node = self._ensure_node(node)
        if not node in self.children:
            if index is None:
                index = len(self.children)
            else:
                if index < 0 or index > len(self.children):
                    raise ValueError("Index out of bounds.")
            self.children.insert(index, node)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_root(self):
        return len(self.parents) == 0

    def is_leaf(self):
        return len(self.children) == 0

    def remove(self, keep_parents=False, keep_children=False):
        """Remove references to self from parents and children.

        Parameters
        ----------
        parent_or_index : Node or int
        """
        if not keep_parents:
            while len(self.parents) > 0:
                self.remove_parent(0)
        if not keep_children:
            for c in self.children.copy():
                c.remove_parent(self)

    def remove_parent(self, parent_or_index):
        """Remove a parent from self and self from parent's children.

        Parameters
        ----------
        parent_or_index : Node or int
        """
        index = parent_or_index
        if isinstance(index, Node):
            for i, p in enumerate(self.parents):
                if p == parent_or_index:
                    index = i
                    break
        if isinstance(index, Node):
            raise Exception("Could not find a parent")
        parent = self.parents[index]
        del self.parents[index]
        parent.children.remove(self)
        return index

    def change_to(self, node, transfer_parents=True, transfer_children=True):
        """Effectively changes self to another node. Reference to self is untouched.

        Parameters
        ----------
        node : Node
            The new Node to change self to.
        transfer_parents : boolean
            Whether to reuse current parents.
        transfer_children : boolean
            Whether to reuse current children, which will also be reset recursively.

        Returns
        -------
        node : Node
            The new node with parents and children associated.
        """
        if transfer_parents:
            parents = self.parents.copy()
            for p in parents:
                self.remove_parent(p)
            node.add_parents(parents)

        if transfer_children:
            children = self.children.copy()
            for c in children:
                index = c.remove_parent(self)
                c.add_parent(node, index=index)
                c.reset(propagate=True)

        return node

    @property
    def ancestors(self):
        _ancestors = self.parents.copy()
        for n in self.parents:
            for m in n.ancestors:
                if m not in _ancestors:
                    _ancestors.append(m)
        return _ancestors

    @property
    def descendants(self):
        _descendants = self.children.copy()
        for n in self.children:
            for m in n.descendants:
                if m not in _descendants:
                    _descendants.append(m)
        return _descendants

    @property
    def component(self):
        return [self] + self.ancestors + self.descendants

    #@property
    #def graph(self):
    #    return Graph(self)

    @property
    def label(self):
        return self.name

    @property
    def neighbours(self):
        return self.children + self.parents

    """Private methods"""

    def _convert_to_node(self, obj, name):
        raise ValueError("No conversion to Node for value {}".format(obj))

    def _ensure_node(self, obj):
        if isinstance(obj, Node):
            return obj
        name = "_{}_{}".format(self.name, str(uuid.uuid4().hex[0:6]))
        return self._convert_to_node(obj, name)

    """Static methods"""

    @staticmethod
    def node_list(nodes):
        if isinstance(nodes, dict):
            nodes = nodes.values()
        elif isinstance(nodes, Node):
            nodes = [nodes]
        return nodes


# TODO: add version number to key so that resets are not confused in dask scheduler
def make_key(name, sl):
    """Makes the dask key for the outputs of nodes

    Parameters
    ----------
    name : string
        name of the output (e.g. node name)
    sl : slice
        data slice that is covered by this output

    Returns
    -------
    a tuple key
    """
    n = slen(sl)
    if n <= 0:
        ValueError("Slice has no length")
    return (name, sl.start, n)


def is_elfi_key(key):
    return isinstance(key, tuple) and len(key) == 3 and isinstance(key[0], str)


def get_key_slice(key):
    """Returns the corresponding slice from 'key'.
    """
    return slice(key[1], key[1] + key[2])


def get_key_name(key):
    return key[0]


def reset_key_slice(key, new_sl):
    """Resets the slice from 'key' to 'new_sl'

    Returns
    -------
    a new key
    """
    return make_key(get_key_name(key), new_sl)


def reset_key_name(key, name):
    """Resets the name from 'key' to 'name'

    Returns
    -------
    a new key
    """
    return make_key(name, get_key_slice(key))


def get_named_item(output, item, name=None):
    """Makes a delayed object by appending "-name" to the output key name

    Parameters
    ----------
    output : delayed node output
    item : str
       item to take from the output
    name : str
       delayed key name (default: item)

    Returns
    -------
    delayed object yielding the item
    """
    name = name or item
    new_key_name = get_key_name(output.key) + '-' + str(name)
    new_key = reset_key_name(output.key, new_key_name)
    return delayed(operator.getitem)(output, item, dask_key_name=new_key)


class ElfiStore:
    """Store interface for Elfi outputs and data.

    All implementations must be able to store the output data. Storing the output
    dict is optional.

    """

    def write(self, output, done_callback=None):
        """Write output or output data to store

        Parameters
        ----------
        output : delayed output
        done_callback : fn(key, result)
           result is either the concrete result or a finished future for the result

        """
        raise NotImplementedError

    def read(self, key):
        """Implementation of this method is optional.

        Parameters
        ----------
        key : output key

        Returns
        -------
        the output result
        """
        raise NotImplementedError

    def read_data(self, node_name, sl):
        """

        Parameters
        ----------
        sl : slice
        node_name : string

        Returns
        -------
        dask.delayed object yielding the data matching the slice with .compute()
        """
        raise NotImplementedError

    def reset(self):
        """Reset the store to the initial state. All results will be cleared.
        """
        raise NotImplementedError


class LocalElfiStore(ElfiStore):
    """Interface for any "local object store".
    """

    def __init__(self):
        self._pending_persisted = defaultdict(lambda: None)

    def _read_data(self, name, sl):
        """Operation for reading from storage.

        Parameters
        ----------
        name : string
            Name of node (all ilmplementations may not use this).
        sl : slice
            Indices for data to return.

        Returns
        -------
        np.ndarray
            Data matching slice, shape: (slice length, ) + data.shape
        """
        raise NotImplementedError

    def _write(self, key, output_result):
        """Operation for writing to storage.

        Parameters
        ----------
        key : tuple
            Dask key matching the write operation.
            Contains both node name (with get_key_name) and
            slice (with get_key_slice).
        output_result : dict
            Operation output dict:
                "data" : data to be stored (np.ndarray)
                         shape should be (slice length, ) + data.shape
        """
        raise NotImplementedError

    def _reset(self):
        """Operation for resetting storage object (optional).
        """
        pass

    def write(self, output, done_callback=None):
        key = output.key
        key_name = get_key_name(key)
        d = env.client().persist(output)
        # We must keep the reference around so that the result is not cleared from memory
        self._pending_persisted[key] = d
        # Take out the underlying future
        future = d.dask[key]
        future.add_done_callback(lambda f: self._post_task(key, f, done_callback))

    def read_data(self, node_name, sl):
        name = node_name + "-data"
        key = make_key(name, sl)
        return delayed(self._read_data(node_name, sl), name=key, pure=True)

    def reset(self):
        self._pending_persisted.clear()
        self._reset()

    # Issue https://github.com/dask/distributed/issues/647
    @gen.coroutine
    def _post_task(self, key, future, done_callback=None):
        res = yield future._result()
        self._write(key, res)
        # Inform that the result is stored
        if done_callback is not None:
            done_callback(key, res)
        # Remove the future reference
        del self._pending_persisted[key]


class LocalDataStore(LocalElfiStore):
    """Implementation for any simple sliceable storage object.

    Object should have following methods:
        __getitem__, __setitem__, __len__

    Examples: numpy array, h5py instance.

    The storage object should have enough space to hold all samples.
    For example, numpy array shape should be at least (n_samples, ) + data.shape.

    The slicing operation must be consistent:
        'obj[sl] = d' must guarantee that 'obj[sl] == d'
        For example, an empty list will not guarantee this, but a pre-allocated will.
    """
    def __init__(self, local_store):
        if not (getattr(local_store, "__getitem__", False) and callable(local_store.__getitem__)):
            raise ValueError("Store object does not implement __getitem__.")
        if not (getattr(local_store, "__setitem__", False) and callable(local_store.__setitem__)):
            raise ValueError("Store object does not implement __setitem__.")
        if not (getattr(local_store, "__len__", False) and callable(local_store.__len__)):
            raise ValueError("Store object does not implement __len__.")
        self._local_store = local_store
        super(LocalDataStore, self).__init__()

    def _read_data(self, name, sl):
        return self._local_store[sl]

    def _write(self, key, output_result):
        sl = get_key_slice(key)
        if len(self._local_store) < sl.stop:
            raise IndexError("No more space on local storage object")
        self._local_store[sl] = output_result["data"]


class MemoryStore(ElfiStore):
    """Cache results in memory of the workers using dask.distributed."""
    def __init__(self):
        self._persisted = defaultdict(lambda: None)

    def write(self, output, done_callback=None):
        key = output.key
        # Persist key to client
        d = env.client().persist(output)
        self._persisted[key] = d

        future = d.dask[key]
        if done_callback is not None:
            future.add_done_callback(lambda f: done_callback(key, f))

    def read(self, key):
        return self._persisted[key].compute()

    def read_data(self, node_name, sl):
        sl = to_slice(sl)
        # TODO: allow arbitrary slices to be taken, now they have to match
        output = [d for key, d in self._persisted.items() if get_key_slice(key) == sl]
        if len(output) == 0:
            raise IndexError("No matching slice found.")
        return get_named_item(output[0], 'data')

    def reset(self):
        self._persisted.clear()


class DelayedOutputCache:
    """Handles a continuous list of delayed outputs for a node.
    """
    def __init__(self, node_name, store=None):
        """

        Parameters
        ----------
        store : None, ElfiStore, string, or sliceable object
            None : means data is not stored.
            ElfiStore derivative : stores data according to specification.
            String identifiers :
                "cache" : Creates a MemoryStore()
            Sliceable object : is converted to LocalDataStore(obj)
                Examples: local numpy array, h5py instance.
                The size of the object must be at least (n_samples, ) + data.shape
                The slicing must be consistent:
                    obj[sl] = d must guarantee that obj[sl] == d
                    For example, an empty list will not guarantee this, but a pre-allocated will.
                See also: LocalDataStore
        """
        self._delayed_outputs = []
        self._stored_mask = []
        self._store = self._prepare_store(store)
        self._node_name = node_name

    def _prepare_store(self, store):
        # Handle local store objects
        if store is None:
            return None
        if isinstance(store, ElfiStore):
            return store
        if type(store) == str:
            if store.lower() == "cache":
                return MemoryStore()
            raise ValueError("Unknown store identifier '{}'".format(store))
        return LocalDataStore(store)

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

    def reset(self):
        del self._delayed_outputs[:]
        del self._stored_mask[:]
        if self._store is not None:
            self._store.reset()

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
                output_data = self._store.read_data(self._node_name, output_sl)
            else:
                output_data = get_named_item(output, 'data')

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


class Operation(Node):
    def __init__(self, name, operation, *parents, store=None):
        """

        Parameters
        ----------
        name : name of the node
        operation : node operation function
        *parents : parents of the nodes
        store : `OutputStore` instance
        """
        super(Operation, self).__init__(name, *parents)
        self.operation = operation

        self._generate_index = 0
        self._delayed_outputs = DelayedOutputCache(name, store)
        self.reset(propagate=False)

    def acquire(self, n, starting=0, batch_size=None):
        """
        Acquires values from the start or from starting index.
        Generates new ones if needed and updates the _generate_index.
        """
        sl = slice(starting, starting+n)
        if self._generate_index < sl.stop:
            self.generate(sl.stop - self._generate_index, batch_size=batch_size)
        return self.get_slice(sl)

    def generate(self, n, batch_size=None, with_values=None):
        """Generate n new values from the node.
        """
        a = self._generate_index
        b = a + n
        batch_size = batch_size or n
        with_values = normalize_data_dict(with_values, n)

        # TODO: with_values cannot be used with already generated values
        # Ensure store is filled up to `b`
        while len(self._delayed_outputs) < b:
            l = len(self._delayed_outputs)
            n_batch = min(b-l, batch_size)
            batch_sl = slice(l, l+n_batch)
            batch_values = None
            if with_values is not None:
                batch_values = {k: v[(l-a):(l-a)+n_batch] for k,v in with_values.items()}
            self.get_slice(batch_sl, with_values=batch_values)

        self._generate_index = b
        return self[slice(a, b)]

    def __getitem__(self, sl):
        sl = to_slice(sl)
        return self._delayed_outputs[sl]

    def get_slice(self, sl, with_values=None):
        """
        This function is ensured to give a slice anywhere (already generated or not)
        Does not update _generate_index
        """
        # TODO: prevent using with_values with already generated values
        # Check if we need to generate new
        if len(self._delayed_outputs) < sl.stop:
            with_values = normalize_data_dict(with_values, sl.stop - len(self._delayed_outputs))
            new_sl = slice(len(self._delayed_outputs), sl.stop)
            new_input = self._create_input_dict(new_sl, with_values=with_values)
            new_output = self._create_delayed_output(new_sl, new_input, with_values)
            self._delayed_outputs.append(new_output)
        return self[sl]

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
        self._generate_index = 0
        self._delayed_outputs.reset()

    def _create_input_dict(self, sl, with_values=None):
        n = sl.stop - sl.start
        input_data = tuple([p.get_slice(sl, with_values) for p in self.parents])
        return {
            "data": input_data,
            "n": n,
            "index": sl.start,
        }

    def _create_delayed_output(self, sl, input_dict, with_values=None):
        """

        Parameters
        ----------
        sl : slice
        input_dict : dict
        with_values : dict {'node_name': np.array}

        Returns
        -------
        out : dask.delayed object
            object.key is (self.name, sl.start, n)

        """
        with_values = with_values or {}
        dask_key_name = make_key(self.name, sl)
        if self.name in with_values:
            # Set the data to with_values
            output = to_output_dict(input_dict, data=with_values[self.name])
            return delayed(output, name=dask_key_name)
        else:
            dinput = delayed(input_dict, pure=True)
            return delayed(self.operation)(dinput,
                                           dask_key_name=dask_key_name)

    def _convert_to_node(self, obj, name):
        return Constant(name, obj)


class Constant(Operation):
    def __init__(self, name, value):
        if type(value) in (tuple, list, np.ndarray):
            self.value = normalize_data(value, len(value))
        else:
            self.value = normalize_data(value, 1)
        v = self.value.copy()
        super(Constant, self).__init__(name, lambda input_dict: {"data": v})


"""Operation mixins add additional functionality to the Operation class.
They do not define the actual operation. They only add keyword arguments.
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
        # Fixme: decide where to set the inference model seed
        self.seed = 0

    def _create_input_dict(self, sl, **kwargs):
        dct = super(RandomStateMixin, self)._create_input_dict(sl, **kwargs)
        dct["random_state"] = self._get_random_state()
        return dct

    def _get_random_state(self):
        i_subs = next(substreams)
        return delayed(get_substream_state, pure=True)(self.seed, i_subs)


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
        observed = self.operation({"data": observed, "n": 1})["data"]
        return observed


"""
ABC specific Operation nodes
"""

def vectorize_simulator(simulator, *input_data, n_sim=1, prng=None):
    """Used to vectorize a sequential simulation operation
    """
    data = None
    for i in range(n_sim):
        inputs = [v[i] for v in input_data]
        d = simulator(*inputs, prng=prng)
        if not isinstance(d, np.ndarray):
            raise ValueError("Simulation operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(d)))
        if data is None:
            data = np.zeros((n_sim,) + d.shape)
        data[i] = d
    return data

# For python simulators using numpy random variables
def simulator_operation(simulator, vectorized, input_dict):
    """Calls the simulator to produce output

    Vectorized simulators
    ---------------------
    Calls the simulator(*vectorized_args, n_sim, prng) to create output.
    Each vectorized argument to simulator is a numpy array with shape[0] == 'n_sim'.
    Simulator should return a numpy array with shape[0] == 'n_sim'.

    Sequential simulators
    ---------------------
    Calls the simulator(*args, prng) 'n_sim' times to create output.
    Each argument to simulator is of the dtype of the original array[i].
    Simulator should return a numpy array.

    Parameters
    ----------
    simulator: function
    vectorized: bool
    input_dict: dict
        "n": number of parallel simulations
        "data": list of args as numpy arrays
    """
    # set the random state
    prng = np.random.RandomState(0)
    prng.set_state(input_dict["random_state"])
    n_sim = input_dict["n"]
    data = simulator(*input_dict["data"], n_sim=n_sim, prng=prng)
    if not isinstance(data, np.ndarray):
        raise ValueError("Simulation operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(data)))
    if data.shape[0] != n_sim or len(data.shape) < 2:
        raise ValueError("Simulation operation output format incorrect." +
                " Expected shape == ({}, ...).".format(n_sim) +
                " Received shape == {}.".format(data.shape))
    return to_output_dict(input_dict, data=data, random_state=prng.get_state())


class Simulator(ObservedMixin, RandomStateMixin, Operation):
    """Simulator node

    Parameters
    ----------
    name: string
    simulator: function
    vectorized: bool
        whether the simulator function is vectorized or not
        see definition of simulator_operation for more information
    """
    def __init__(self, name, simulator, *args, vectorized=True, **kwargs):
        if vectorized is False:
            simulator = partial(vectorize_simulator, simulator)
        operation = partial(simulator_operation, simulator, vectorized)
        super(Simulator, self).__init__(name, operation, *args, **kwargs)


def vectorize_summary(summary, *input_data):
    """Used to vectorize a sequential summary operation
    """
    data = None
    # TODO: should summary operations also get n_sim as parameter?
    n_sim = input_data[0].shape[0]
    for i in range(n_sim):
        inputs = [v[i] for v in input_data]
        d = summary(*inputs)
        if not isinstance(d, np.ndarray):
            raise ValueError("Summary operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(d)))
        if data is None:
            data = np.zeros((n_sim,) + d.shape)
        data[i] = d
    return data

def summary_operation(operation, input):
    data = operation(*input["data"])
    vec_len = input["n"]
    if not isinstance(data, np.ndarray):
        raise ValueError("Summary operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(data)))
    if data.shape[0] != vec_len or len(data.shape) < 2:
        raise ValueError("Summary operation output format incorrect." +
                " Expected shape == ({}, ...).".format(vec_len) +
                " Received shape == {}.".format(data.shape))
    return to_output_dict(input, data=data)


class Summary(ObservedMixin, Operation):
    def __init__(self, name, summary, *args, vectorized=True, **kwargs):
        if vectorized is False:
            summary = partial(vectorize_summary, summary)
        operation = partial(summary_operation, summary)
        super(Summary, self).__init__(name, operation, *args, **kwargs)


def vectorize_discrepancy(discrepancy, x, y):
    """Used to vectorize a sequential discrepancy operation
    """
    # TODO: should discrepancy operations also get n_sim as parameter?
    n_sim = x[0].shape[0]
    data = np.zeros((n_sim, 1))
    for i in range(n_sim):
        xi = tuple([v[i] for v in x])
        yi = tuple([v[0] for v in y])
        d = discrepancy(x, y)
        if not isinstance(d, np.ndarray):
            raise ValueError("Discrepancy operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(d)))
        if d.shape != (1,):
            raise ValueError("Discrepancy operation output format incorrect." +
                " Expected shape == (1,)." +
                " Received shape == {}.".format(data.shape))
        data[i] = d
    return data

def discrepancy_operation(operation, input):
    data = operation(input["data"], input["observed"])
    vec_len = input["n"]
    if not isinstance(data, np.ndarray):
        raise ValueError("Discrepancy operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(data)))
    if data.shape != (vec_len, 1):
        raise ValueError("Discrepancy operation output format incorrect." +
                " Expected shape == ({}, 1).".format(vec_len) +
                " Received shape == {}.".format(data.shape))
    return to_output_dict(input, data=data)


class Discrepancy(Operation):
    """The operation input has a tuple of data and tuple of observed
    """
    def __init__(self, name, discrepancy, *args, vectorized=True, **kwargs):
        if vectorized is False:
            discrepancy = partial(vectorize_discrepancy, discrepancy)
        operation = partial(discrepancy_operation, discrepancy)
        super(Discrepancy, self).__init__(name, operation, *args, **kwargs)

    def _create_input_dict(self, sl, **kwargs):
        dct = super(Discrepancy, self)._create_input_dict(sl, **kwargs)
        dct["observed"] = observed = tuple([p.observed for p in self.parents])
        return dct


def threshold_operation(threshold, input):
    data = input['data'][0] < threshold
    return to_output_dict(input, data=data)


class Threshold(Operation):
    def __init__(self, name, threshold, *args, **kwargs):
        operation = partial(threshold_operation, threshold)
        super(Threshold, self).__init__(name, operation, *args, **kwargs)


"""Other functions
"""

# Not used?
#def fixed_expand(n, fixed_value):
#    """Creates a new axis 0 (or dimension) along which the value is repeated
#    """
#    return np.repeat(fixed_value[np.newaxis,:], n, axis=0)
#
# class Graph(object):
#     """A container for the graphical model"""
#     def __init__(self, anchor_node=None):
#         self.anchor_node = anchor_node
#
#     @property
#     def nodes(self):
#         return self.anchor_node.component
#
#     def sample(self, n, parameters=None, threshold=None, observe=None):
#         raise NotImplementedError
#
#     def posterior(self, N):
#         raise NotImplementedError
#
#     def reset(self):
#         data_nodes = self.find_nodes(Data)
#         for n in data_nodes:
#             n.reset()
#
#     def find_nodes(self, node_class=Node):
#         nodes = []
#         for n in self.nodes:
#             if isinstance(n, node_class):
#                 nodes.append(n)
#         return nodes
#
#     def __getitem__(self, key):
#         for n in self.nodes:
#             if n.name == key:
#                 return n
#         raise IndexError
#
#     def __getattr__(self, item):
#         for n in self.nodes:
#             if n.name == item:
#                 return n
#         raise AttributeError
#
#     def plot(self, graph_name=None, filename=None, label=None):
#         from graphviz import Digraph
#         G = Digraph(graph_name, filename=filename)
#
#         observed = {"shape": "box", "fillcolor": "grey", "style": "filled"}
#
#         # add nodes
#         for n in self.nodes:
#             if isinstance(n, Fixed):
#                 G.node(n.name, xlabel=n.label, shape="point")
#             elif hasattr(n, "observed") and n.observed is not None:
#                 G.node(n.name, label=n.label, **observed)
#             # elif isinstance(n, Discrepancy) or isinstance(n, Threshold):
#             #     G.node(n.name, label=n.label, **observed)
#             else:
#                 G.node(n.name, label=n.label, shape="doublecircle",
#                        fillcolor="deepskyblue3",
#                        style="filled")
#
#         # add edges
#         edges = []
#         for n in self.nodes:
#             for c in n.children:
#                 if (n.name, c.name) not in edges:
#                     edges.append((n.name, c.name))
#                     G.edge(n.name, c.name)
#             for p in n.parents:
#                 if (p.name, n.name) not in edges:
#                     edges.append((p.name, n.name))
#                     G.edge(p.name, n.name)
#
#         if label is not None:
#             G.body.append("label=" + "\"" + label + "\"")
#
#         return G
#
#     """Properties"""
#
#     @property
#     def thresholds(self):
#         return self.find_nodes(node_class=Threshold)
#
#     @property
#     def discrepancies(self):
#         return self.find_nodes(node_class=Discrepancy)
#
#     @property
#     def simulators(self):
#         return [node for node in self.nodes if isinstance(node, Simulator)]
#
#     @property
#     def priors(self):
#         raise NotImplementedError
#         #Implementation wrong, prior have Value nodes as hyperparameters
#         # priors = self.find_nodes(node_class=Stochastic)
#         # priors = {n for n in priors if n.is_root()}
#         # return priors

if __name__ == "__main__":
    import doctest
    doctest.testmod()
