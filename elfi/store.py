import logging
import os
import io
import shutil
import pickle

import numpy as np
import numpy.lib.format as npformat

logger = logging.getLogger(__name__)


def _get_default_prefix():
    return 'pools'


class OutputPool:
    """Store node outputs to dictionary-like stores.

    The default store is a Python dictionary.

    Notes
    -----
    Saving the store requires that all the stores are pickleable.

    Arbitrary objects that support simple array indexing can be used as stores by using
    the `elfi.store.ArrayObjectStore` class.

    See the `elfi.store.StoreBase` interfaces if you wish to implement your own ELFI
    compatible store. Basically any object that fulfills the Pythons dictionary
    api will work as an store in the pool.

    """

    def __init__(self, outputs=None, name=None, prefix=None):
        """

        Depending on the algorithm, some of these values may be reused
        after making some changes to `ElfiModel` thus speeding up the inference 
        significantly. For instance, if all the simulations are stored in Rejection 
        sampling, one can change the summaries and distances without having to rerun
        the simulator.
        
        Parameters
        ----------
        outputs : list, dict, optional
            list of node names which to store or a dictionary with existing stores. The
            stores are created on demand.
        name : str, optional
            Name of the pool. Used to open a saved pool from disk.
        prefix : str, optional
            Path to directory under which `elfi.ArrayPool` will place its folder.
            Default is a relative path ./pools.
            
        Returns
        -------
        instance : OutputPool
        """

        if outputs is None:
            stores = {}
        elif isinstance(outputs, dict):
            stores = outputs
        else:
            stores = dict.fromkeys(outputs)

        self.stores = stores

        # Context information
        self.batch_size = None
        self.seed = None

        self.name = name
        self.prefix = prefix or _get_default_prefix()

        if self.path and os.path.exists(self.path):
            raise ValueError("A pool with this name already exists in {}. You can use "
                             "OutputPool.open() to open it.".format(self.prefix))

    @property
    def output_names(self):
        return list(self.stores.keys())

    @property
    def context_set(self):
        return self.seed is not None and self.batch_size is not None

    def set_context(self, context):
        """Sets the context of the pool.

        The pool needs to know the batch_size and the seed.

        Notes
        -----
        Also sets the name of the pool if not set already.
        
        Parameters
        ----------
        context : elfi.ComputationContext

        Returns
        -------
        None
        """
        if self.context_set:
            raise ValueError('Context is already set')

        self.batch_size = context.batch_size
        self.seed = context.seed

        if self.name is None:
            self.name = "{}_{}".format(self.__class__.__name__.lower(), self.seed)

    def get_batch(self, batch_index, output_names=None):
        """Returns a batch from the stores of the pool.
        
        Parameters
        ----------
        batch_index : int
        output_names : list
            which outputs to include to the batch

        Returns
        -------
        batch : dict
        """

        output_names = output_names or self.output_names
        batch = dict()
        for output in output_names:
            store = self.stores[output]
            if store is None:
                continue
            if batch_index in store:
                batch[output] = store[batch_index]
        return batch

    def add_batch(self, batch, batch_index):
        """Adds the outputs from the batch to their stores."""
        for node, values in batch.items():
            if node not in self.stores:
                continue
            store = self._get_store_for(node)

            # Do not add again. The output should be the same.
            if batch_index in store:
                continue

            store[batch_index] = values

    def remove_batch(self, batch_index):
        """Removes the batch from all the stores."""
        for store in self.stores.values():
            if batch_index in store:
                del store[batch_index]

    def has_store(self, node):
        return node in self.stores

    def get_store(self, node):
        return self.stores[node]

    def add_store(self, node, store=None):
        """Adds a store object for the node.

        Parameters
        ----------
        node : str
        store : dict, StoreBase, optional

        Returns
        -------
        None

        """
        if node in self.stores and self.stores[node] is not None:
            raise ValueError("Store for '{}' already exists".format(node))

        store = store or self._make_store_for(node)
        self.stores[node] = store

    def remove_store(self, node):
        """Removes a store from the pool

        Parameters
        ----------
        node : str

        Returns
        -------
        store
            The removed store
        """
        store = self.stores.pop(node)
        return store

    def _get_store_for(self, node):
        """Gets or makes a store."""
        if self.stores[node] is None:
            self.stores[node] = self._make_store_for(node)
        return self.stores[node]

    def _make_store_for(self, node):
        """Make a default store for a node

        All the default stores will be created through this method.
        """
        return {}

    def __len__(self):
        """Largest batch index in any of the stores"""
        l = 0
        for output, store in self.stores.items():
            if store is None:
                continue
            l = max(l, len(store))
        return l

    def __getitem__(self, batch_index):
        """Return the batch"""
        return self.get_batch(batch_index)

    def __setitem__(self, batch_index, batch):
        return self.add_batch(batch, batch_index)

    def __contains__(self, batch_index):
        return len(self) > batch_index

    def clear(self):
        """Removes all data from the stores"""
        for store in self.stores.values():
            store.clear()

    def save(self):
        """Save the pool to disk.

        This will use pickle to store the pool under self.path.
        """
        if not self.context_set:
            raise ValueError("Pool context is not set, cannot save. Please see the "
                             "set_context method.")

        os.makedirs(self.path, exist_ok=True)

        # Pickle the stores separately
        for node, store in self.stores.items():
            filename = os.path.join(self.path, node + '.pkl')
            try:
                pickle.dump(store, open(filename, 'wb'))
            except:
                raise IOError('Failed to pickle the store for node {}, please check that '
                              'it is pickleable or remove it before saving.'.format(node))

        # Save the pool itself with stores replaced with Nones
        stores = self.stores
        self.stores = dict.fromkeys(stores.keys())
        filename = os.path.join(self.path, self._get_pkl_name())
        pickle.dump(self, open(filename, "wb"))
        # Restore the original to the object
        self.stores = stores

    def close(self):
        """Save and close the stores that support it.

        The pool will not be usable afterwards."""
        self.save()

        for store in self.stores.values():
            if hasattr(store, 'close'):
                store.close()

    def flush(self):
        """Flushes all data from the stores.

        If the store does not support flushing, does nothing.
        """
        for store in self.stores.values():
            if hasattr(store, 'flush'):
                store.flush()

    def delete(self):
        """Remove all persisted data from disk."""
        for store in self.stores.values():
            if hasattr(store, 'close'):
                store.close()

        if self.path is None:
            return
        elif not os.path.exists(self.path):
            return

        shutil.rmtree(self.path)

    @classmethod
    def open(cls, name, prefix=None):
        """Open a closed or saved ArrayPool from disk.

        Parameters
        ----------
        name : str
        prefix : str, optional

        Returns
        -------

        """
        prefix = prefix or _get_default_prefix()
        path = cls._make_path(name, prefix)
        filename = os.path.join(path, cls._get_pkl_name())

        pool = pickle.load(open(filename, "rb"))

        # Load the stores. Change the working directory temporarily so that pickled stores
        # can find their data dependencies even if the folder has been renamed.
        cwd = os.getcwd()
        os.chdir(path)
        for node in list(pool.stores.keys()):
            filename = node + '.pkl'
            try:
                store = pickle.load(open(filename, 'rb'))
            except Exception as e:
                logger.warning('Failed to load the store for node {}. Reason: {}'
                               .format(node, str(e)))
                del pool.stores[node]
                continue
            pool.stores[node] = store
        os.chdir(cwd)

        # Update the name and prefix in case the pool folder was moved
        pool.name = name
        pool.prefix = prefix
        return pool

    @classmethod
    def _make_path(cls, name, prefix):
        return os.path.join(prefix, name)

    @property
    def path(self):
        if self.name is None:
            return None

        return self._make_path(self.name, self.prefix)

    @classmethod
    def _get_pkl_name(cls):
        return 'pool.pkl'


class ArrayPool(OutputPool):
    """OutputPool that uses binary .npy files as default stores.

    The default store medium for output data is a NumPy binary `.npy` file for NumPy
    array data. You can however also add other types of stores as well.

    Notes
    -----
    The default store is elfi.store.NpyStore that uses NpyArrays as stores. The latter is
    a wrapper over NumPy .npy binary file for array data.

    """

    def _make_store_for(self, node):
        if not self.context_set:
            raise ValueError('ArrayPool has no context set')

        # Make the directory for the array pools
        os.makedirs(self.path, exist_ok=True)

        filename = os.path.join(self.path, node)
        return NpyStore(filename, self.batch_size)

    def load_npy_file(self, node, n_batches=None):
        """

        Parameters
        ----------
        node : str
            The node.npy file must be in self.path
        n_batches : int, optional
            How many batches to load. Default is as many as the array has length.

        """
        if not self.context_set:
            raise ValueError('ArrayPool has no context set')

        if node in self.stores and self.stores[node] is not None:
            raise ValueError('Store already exists!')

        store = self._make_store_for(node)
        if n_batches is None:
            if len(store.array) % self.batch_size != 0:
                raise ValueError('The array length is not divisible by the batch size')
            n_batches = len(store.array) // self.batch_size
        else:
            store.array.truncate(n_batches*self.batch_size)

        store.n_batches = n_batches

        self.add_store(node, store)


class StoreBase:
    """Base class for output stores for the pools.

    Stores store the outputs of a single node in ElfiModel. This is a subset of the
    Python dictionary api.

    Notes
    -----
    Any dictionary like object will work directly as an ELFI store.

    """
    def __getitem__(self, batch_index):
        raise NotImplementedError

    def __setitem__(self, batch_index, data):
        raise NotImplementedError

    def __delitem__(self, batch_index):
        raise NotImplementedError

    def __contains__(self, batch_index):
        raise NotImplementedError

    def __len__(self):
        """Number of batches in the store"""
        raise NotImplementedError

    def clear(self):
        """Remove all batches from the store"""
        raise NotImplementedError

    def close(self):
        """Close the store.

        Optional method. Useful for closing i.e. file streams"""
        pass

    def flush(self):
        """Flush the store

        Optional to implement.
        """
        pass


# TODO: add mask for missing items. It should replace the use of `n_batches`.
#       This should make it possible to also append further than directly to the end
#       of current n_batches index.
class ArrayStore(StoreBase):
    """Convert any array object to ELFI store to be used within a pool.

    This class is intended to make it easy to use objects that support array indexing
    as outputs stores for nodes.
    """
    def __init__(self, array, batch_size, n_batches=0):
        """

        Parameters
        ----------
        array
            Any array like object supporting Python list indexing
        batch_size : int
            Size of a batch of data
        n_batches : int
            When using pre allocated arrays, this keeps track of the number of batches
            currently stored to the array.
        """
        self.array = array
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __getitem__(self, batch_index):
        sl = self._to_slice(batch_index)
        return self.array[sl]

    def __setitem__(self, batch_index, data):
        if batch_index > self.n_batches:
            raise IndexError("Appending further than to the end of the store array is "
                             "currently not supported.")

        sl = self._to_slice(batch_index)
        if sl.stop > len(self.array):
            raise IndexError("There is not enough space left in the store array.")

        self.array[sl] = data
        self.n_batches += 1

    def __contains__(self, batch_index):
        return batch_index < self.n_batches

    def __delitem__(self, batch_index):
        if batch_index not in self:
            raise IndexError("Cannot remove, batch index {} is not in the array"
                             .format(batch_index))
        elif batch_index != self.n_batches:
            raise IndexError("Removing batches from the middle of the store array is "
                             "currently not supported.")

        self.n_batches -= 1

    def __len__(self):
        return self.n_batches

    def _to_slice(self, batch_index):
        a = self.batch_size*batch_index
        return slice(a, a + self.batch_size)

    def clear(self):
        if hasattr(self.array, 'clear'):
            self.array.clear()
        self.n_batches = 0

    def flush(self):
        if hasattr(self.array, 'flush'):
            self.array.flush()

    def close(self):
        if hasattr(self.array, 'close'):
            self.array.close()


class NpyStore(ArrayStore):
    """Store data to binary .npy files

    Uses the NpyArray objects as an array store.
    """

    def __init__(self, filename, batch_size):
        """

        Parameters
        ----------
        filename : str
            Path to the file, relative or absolute
        batch_size
        """
        array = NpyArray(filename)
        super(NpyStore, self).__init__(array, batch_size)

    def __setitem__(self, batch_index, data):
        sl = self._to_slice(batch_index)
        # NpyArray supports appending
        if batch_index == self.n_batches and sl.start == len(self.array):
            self.array.append(data)
            self.n_batches += 1
            return

        super(NpyStore, self).__setitem__(batch_index, data)

    def __delitem__(self, batch_index):
        super(NpyStore, self).__delitem__(batch_index)
        sl = self._to_slice(batch_index)
        self.array.truncate(sl.start)


class NpyArray:
    """

    Notes
    -----
    - Supports only binary files.
    - Supports only .npy version 2.0
    - See numpy.lib.npformat for documentation of the .npy format """

    MAX_SHAPE_LEN = 2**64

    # Version 2.0 header prefix length
    HEADER_DATA_OFFSET = 12
    HEADER_DATA_SIZE_OFFSET = 8

    def __init__(self, filename, array=None, truncate=False):
        """

        Parameters
        ----------
        filename : str
            File name
        array : ndarray, optional
            Initial array
        truncate : bool
            Whether to truncate the file or not
        """

        self.header_length = None
        self.itemsize = None

        # Header data fields
        self.shape = None
        self.fortran_order = False
        self.dtype = None

        # The header bytes must be prepared in advance, because there is an import in
        # `numpy.lib.format._write_array_header` (1.11.3) that fails if the program is
        # being closed on exception and would corrupt the .npy file.
        self._header_bytes_to_write = None

        if filename[-4:] != '.npy':
            filename += '.npy'
        self.filename = filename

        self.fs = None
        if truncate is False and os.path.exists(self.filename):
            self.fs = open(self.filename, 'r+b')
            self._init_from_file_header()
        else:
            self.fs = open(self.filename, 'w+b')

        if array is not None:
            self.append(array)
            self.flush()

    def __getitem__(self, sl):
        if self.header_length is None:
            raise IndexError()
        order = 'F' if self.fortran_order else 'C'
        # TODO: do not recreate if nothing has changed
        mmap = np.memmap(self.fs, dtype=self.dtype, shape=self.shape,
                         offset=self.header_length, order=order)
        return mmap[sl]

    def __setitem__(self, sl, value):
        if self.header_length is None:
            raise IndexError()
        order = 'F' if self.fortran_order else 'C'
        mmap = np.memmap(self.fs, dtype=self.dtype, shape=self.shape,
                         offset=self.header_length, order=order)
        mmap[sl] = value

    def __len__(self):
        return self.shape[0] if self.shape else 0

    @property
    def size(self):
        return np.prod(self.shape)

    def append(self, array):
        """Append data from array to self."""
        if self.closed:
            raise ValueError('Array is not opened.')

        if not self.initialized:
            self._init_from_array(array)

        if array.shape[1:] != self.shape[1:]:
            raise ValueError("Appended array is of different shape")
        elif array.dtype != self.dtype:
            raise ValueError("Appended array is of different dtype")

        # Append new data
        self.fs.seek(0, 2)
        self.fs.write(array.tobytes('C'))
        self.shape = (self.shape[0] + len(array),) + self.shape[1:]

        # Only prepare the header bytes, need to be flushed to take effect
        self._prepare_header_data()

    def _init_from_file_header(self):
        """Initialize the object from an existing file"""
        self.fs.seek(self.HEADER_DATA_SIZE_OFFSET)
        try:
            self.shape, fortran_order, self.dtype = \
                npformat.read_array_header_2_0(self.fs)
        except ValueError:
            raise ValueError('Npy file header is not 2.0 format. Please initialize with '
                             'preloaded array object instead.')
        self.header_length = self.fs.tell()

        if fortran_order:
            raise ValueError('Column major (Fortran-style) files are not supported. Please'
                             'translate if first to row major (C-style).')

        # Determine itemsize
        shape = (0,) + self.shape[1:]
        self.itemsize = np.empty(shape=shape, dtype=self.dtype).itemsize

    def _init_from_array(self, array):
        """Initialize the object from an array.

        Sets the the header_length so large that it is possible to append to the array.

        Returns
        -------
        h_bytes : io.BytesIO
            Contains the oversized header bytes

        """
        self.shape = (0,) + array.shape[1:]
        self.dtype = array.dtype
        self.itemsize = array.itemsize

        # Read header data from array and set modify it to be large for the length
        # 1_0 is the same for 2_0
        d = npformat.header_data_from_array_1_0(array)
        d['shape'] = (self.MAX_SHAPE_LEN,) + d['shape'][1:]
        d['fortran_order'] = False

        # Write a prefix for a very long array to make it large enough for appending new
        # data
        h_bytes = io.BytesIO()
        npformat.write_array_header_2_0(h_bytes, d)
        self.header_length = h_bytes.tell()

        # Write header prefix to file
        self.fs.seek(0)
        h_bytes.seek(0)
        self.fs.write(h_bytes.read(self.HEADER_DATA_OFFSET))

        # Write header data for the zero length to make it a valid file
        self._prepare_header_data()
        self._write_header_data()

    def truncate(self, length=0):
        """Truncates the array to the specified length

        Parameters
        ----------
        length : int
            Length (=`shape[0]`) of the array to truncate to. Default 0.

        Returns
        -------

        """
        if self.fs is None:
            raise ValueError('Array has been deleted')
        elif self.fs.closed:
            raise ValueError('Array has been closed')

        # Reset length
        self.shape = (length,) + self.shape[1:]
        self._prepare_header_data()
        self._write_header_data()

        self.fs.seek(self.header_length + self.size*self.itemsize)
        self.fs.truncate()

    def close(self):
        if self.initialized:
            self._write_header_data()
            self.fs.close()

    def clear(self):
        self.truncate(0)

    def delete(self):
        """Removes the file and invalidates this array"""
        if self.deleted:
            return
        name = self.fs.name
        self.close()
        os.remove(name)
        self.fs = None
        self.header_length = None

    def flush(self):
        self._write_header_data()
        self.fs.flush()

    def __del__(self):
        self.close()

    def _prepare_header_data(self):
        # Make header data
        d = {
            'shape': self.shape,
            'fortran_order': self.fortran_order,
            'descr': npformat.dtype_to_descr(self.dtype)
        }

        h_bytes = io.BytesIO()
        npformat.write_array_header_2_0(h_bytes, d)

        # Pad the end of the header
        fill_len = self.header_length - h_bytes.tell()
        if fill_len < 0:
            raise OverflowError("File {} cannot be appended. The header is too short.".
                                format(self.filename))
        elif fill_len > 0:
            h_bytes.write(b'\x20' * fill_len)

        h_bytes.seek(0)
        self._header_bytes_to_write = h_bytes.read()

    def _write_header_data(self):
        if not self._header_bytes_to_write:
            return

        # Rewrite header data
        self.fs.seek(self.HEADER_DATA_OFFSET)
        h_bytes = self._header_bytes_to_write[self.HEADER_DATA_OFFSET:]
        self.fs.write(h_bytes)

        # Flag bytes off as they are now written
        self._header_bytes_to_write = None

    @property
    def deleted(self):
        return self.fs is None

    @property
    def closed(self):
        return self.deleted or self.fs.closed

    @property
    def initialized(self):
        return (not self.closed) and (self.header_length is not None)

    def __getstate__(self):
        if not self.fs.closed:
            self.flush()
        return {'filename': self.filename}

    def __setstate__(self, state):
        filename = state.pop('filename')
        basename = os.path.basename(filename)
        if os.path.exists(filename):
            self.__init__(filename)
        elif os.path.exists(basename):
            self.__init__(basename)
        else:
            self.fs = None
            raise FileNotFoundError('Could not find the file {}'.format(filename))


