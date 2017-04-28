import os
import io
import shutil

import numpy as np
import numpy.lib.format as npformat


class OutputPool:
    """Allows storing outputs to different stores."""

    def __init__(self, outputs=None):
        self.output_stores = dict()
        outputs = outputs or {}
        for output in outputs:
            self.output_stores[output] = dict()

        self.batch_size = None
        self.seed = None

    def init_context(self, context):
        self.batch_size = context.batch_size
        self.seed = context.seed

    def get_batch(self, batch_index, outputs=None):
        outputs = outputs or self.outputs
        batch = dict()
        for output in outputs:
            store = self.output_stores[output]
            if batch_index in store:
                batch[output] = store[batch_index]
        return batch

    def __getitem__(self, node):
        return self.output_stores[node]

    def add_batch(self, batch, batch_index):
        for node, store in self.output_stores.items():
            if node not in batch:
                continue
            store[batch_index] = batch[node]

    def add_store(self, name, store=None):
        store = store or {}
        self.output_stores[name] = store

    def remove_store(self, name):
        """Removes a store from the pool

        Parameters
        ----------
        name : str
            Store name

        Returns
        -------
        store
            The removed store
        """
        store = self.output_stores.pop(name)
        return store

    def __setitem__(self, node, store):
        self.output_stores[node] = store

    def __contains__(self, node):
        return node in self.output_stores

    def clear(self):
        """Removes all data from the pool stores"""
        for store in self.output_stores.values():
            store.clear()

    @property
    def outputs(self):
        return self.output_stores.keys()


class ArrayPool(OutputPool):

    def __init__(self, outputs, name='arraypool', basepath=None):
        super(ArrayPool, self).__init__(outputs)

        self.name = name
        self.basepath = basepath or os.path.join(os.path.expanduser('~'), '.elfi')
        os.makedirs(self.basepath, exist_ok=True)

    def init_context(self, context):
        super(ArrayPool, self).init_context(context)

        os.makedirs(self.path)

        # Create the arrays and replace the output dicts with arrays
        for output in self.outputs:
            filename = os.path.join(self.path, output)
            array = NpyPersistedArray(filename)
            self.output_stores[output] = BatchArrayStore(array, self.batch_size)

    @property
    def path(self):
        if self.seed is None:
            raise ValueError('Pool must be initialized with a context (pool.init_context)')
        return os.path.join(self.basepath, self.name, str(self.seed))

    def delete(self):
        try:
            path = self.path
        except:
            # Pool was not initialized
            return
        self.close()
        shutil.rmtree(path)

    def close(self):
        """Closes NpyPersistedArrays"""
        for store in self.output_stores.values():
            if isinstance(store, BatchArrayStore) and hasattr(store.array, 'close'):
                store.array.close()

    def flush(self):
        """Flushes NpyPersistedArrays"""
        for store in self.output_stores.values():
            if isinstance(store, BatchArrayStore) and hasattr(store.array, 'flush'):
                store.array.flush()


class BatchStore:
    """Stores batches for a single node"""
    def __getitem__(self, batch_index):
        raise NotImplementedError

    def __setitem__(self, batch_index, data):
        raise NotImplementedError

    def __contains__(self, batch_index):
        raise NotImplementedError

    def __len__(self):
        """Number of batches in the store"""
        raise NotImplementedError

    def clear(self):
        """Remove all batches from the store"""
        raise NotImplementedError


class BatchArrayStore(BatchStore):
    """Helper class to handle arrays as batch dictionaries"""
    def __init__(self, array, batch_size):
        self.array = array
        self.batch_size = batch_size

    def __contains__(self, batch_index):
        # TODO: implement a mask
        b = self._to_slice(batch_index).stop
        return b <= len(self.array)

    def __getitem__(self, batch_index):
        sl = self._to_slice(batch_index)
        return self.array[sl]

    def __setitem__(self, batch_index, data):
        sl = self._to_slice(batch_index)
        if batch_index in self:
            self[sl] = data
        elif sl.start == len(self.array):
            # TODO: allow appending further than directly to the end
            if hasattr(self.array, 'append'):
                self.array.append(data)
        else:
            raise ValueError('Cannot append to array')

    def __len__(self):
        return int(len(self.array)/self.batch_size)

    def _to_slice(self, batch_index):
        a = self.batch_size*batch_index
        return slice(a, a + self.batch_size)

    def clear(self):
        if hasattr(self.array, 'clear'):
            self.array.clear()


class NpyPersistedArray:
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

    def __init__(self, name, array=None, truncate=False):
        """

        Parameters
        ----------
        name : str
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

        if name[-4:] != '.npy':
            name += '.npy'
        self.name = name

        self.fs = None
        if truncate is False and os.path.exists(self.name):
            self.fs = open(self.name, 'r+b')
            self._init_from_file_header()
        else:
            self.fs = open(self.name, 'w+b')

        if array:
            self.append(array)
            self.flush()

    def __getitem__(self, item):
        if self.header_length is None:
            raise IndexError()
        order = 'F' if self.fortran_order else 'C'
        # TODO: do not recreate if nothing has changed
        mmap = np.memmap(self.fs, dtype=self.dtype, shape=self.shape,
                         offset=self.header_length, order=order)
        return mmap[item]

    def __len__(self):
        return self.shape[0] if self.shape else 0

    @property
    def size(self):
        return np.prod(self.shape)

    def append(self, array):
        """Append data from array to self."""
        if self.fs is None:
            raise ValueError('Array has been deleted')

        if self.header_length is None:
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
        """Initialize the object from existing file"""
        self.fs.seek(self.HEADER_DATA_SIZE_OFFSET)
        self.shape, fortran_order, self.dtype = npformat.read_array_header_2_0(
            self.fs)
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
        if self.header_length:
            self._write_header_data()
            self.fs.close()

    def clear(self):
        self.truncate(0)

    def delete(self):
        """Removes the file and invalidates this array"""
        if not self.fs:
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
                                format(self.name))
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
