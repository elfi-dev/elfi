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

    def add_batch(self, batch_index, batch):
        for node, store in self.output_stores.items():
            if node not in batch:
                continue
            store[batch_index] = batch[node]

    def add_store(self, name, store=None):
        store = store or {}
        self.output_stores[name] = store

    def __setitem__(self, node, store):
        self.output_stores[node] = store

    def __contains__(self, node):
        return node in self.output_stores

    def init_context(self, context):
        self.batch_size = context.batch_size
        self.seed = context.seed

    def destroy(self):
        pass

    @property
    def outputs(self):
        return self.output_stores.keys()


class ArrayPool(OutputPool):

    def __init__(self, outputs, name='default', basepath=None):
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
            raise ValueError('Pool must be initialized with a context (pool.set_context)')
        return os.path.join(self.basepath, self.name, str(self.seed))

    def destroy(self):
        for store in self.output_stores.values():
            store.array.close()

        try:
            path = self.path
        except:
            return

        shutil.rmtree(path)


# TODO: add sqlite3 store, array store (i.e. make a batch interface for them)


class BatchStore:
    """Stores batches for a single node"""
    def __getitem__(self, batch_index):
        raise NotImplementedError

    def __setitem__(self, batch_index, data):
        raise NotImplementedError

    def __contains__(self, batch_index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class BatchArrayStore(BatchStore):
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
        return len(self.array)

    def _to_slice(self, batch_index):
        a = self.batch_size*batch_index
        return slice(a, a + self.batch_size)


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

    def __init__(self, filename, array=None, mode='a'):
        """

        Parameters
        ----------
        filename
        mode : str
           w : truncate and append data to the end
           a : append data to the end
        """

        self.header_length = None

        # Header data fields
        self.shape = None
        self.fortran_order = False
        self.dtype = None
        # The header bytes must be prepared in advance, because there is an import in
        # numpy.lib.format._write_array_header (1.11.3) that fails if the program is being
        # closed on exception and would corrupt the .npy file.
        self._header_bytes = None

        if filename[-4:] != '.npy':
            filename += '.npy'
        self.filename = filename

        self.fs = None
        if mode == 'a' and os.path.exists(self.filename):
            self.fs = open(self.filename, 'r+b')
            self._read_header()
        else:
            self.fs = open(self.filename, 'w+b')

        if array:
            self.append(array)

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

    def append(self, array):
        """Append to the array in place"""
        if self.header_length is None:
            self._init_file(array)
            return

        if array.shape[1:] != self.shape[1:]:
            raise ValueError("Appended array is of different shape")
        elif array.dtype != self.dtype:
            raise ValueError("Appended array is of different dtype")

        # Append new data
        self.fs.seek(0, 2)
        self.fs.write(array.tobytes('C'))
        self.shape = (self.shape[0] + len(array),) + self.shape[1:]

        self._prepare_header_bytes()

    def _read_header(self):
        self.fs.seek(self.HEADER_DATA_SIZE_OFFSET)
        self.shape, self.fortran_order, self.dtype = npformat.read_array_header_2_0(
            self.fs)
        self.header_length = self.fs.tell()

    def _init_file(self, array):
        """Sets a large header that allows the filesize to grow very large"""
        fs = self.fs

        # Read header data from array
        # 1_0 is the same for 2_0
        d = npformat.header_data_from_array_1_0(array)
        self.shape = array.shape
        self.fortran_order = d['fortran_order']
        self.dtype = array.dtype

        # Write a fake over sized header to make it large enough for new data
        h_bytes = io.BytesIO()
        d['shape'] = (self.MAX_SHAPE_LEN,) + d['shape'][1:]
        npformat.write_array_header_2_0(h_bytes, d)
        self.header_length = h_bytes.tell()

        # Write header prefix
        fs.seek(0)
        h_bytes.seek(0)
        fs.write(h_bytes.read(self.HEADER_DATA_OFFSET))

        # Write correct header data
        self._prepare_header_bytes()
        self._write_header_data()
        pos = self.fs.tell()

        # Pad
        fill_len = self.header_length - pos
        if fill_len:
            fs.write(b'\x20' * fill_len)

        # Write data
        fs.write(array.tobytes('C'))

    def close(self):
        if self.header_length:
            self._write_header_data()
            self.fs.close()

    def flush(self):
        self._write_header_data()
        self.fs.flush()

    def __del__(self):
        self.close()

    def _prepare_header_bytes(self):
        # Make header data
        d = {
            'shape': self.shape,
            'fortran_order': self.fortran_order,
            'descr': npformat.dtype_to_descr(self.dtype)
        }

        h_bytes = io.BytesIO()
        npformat.write_array_header_2_0(h_bytes, d)
        h_bytes.seek(0)
        self._header_bytes = h_bytes.read()

    def _write_header_data(self):
        if not self._header_bytes:
            # Nothing to write
            return

        if len(self._header_bytes) > self.header_length:
            raise OverflowError("File {} cannot be appended. The header is too short.".
                                format(self.filename))

        # Rewrite header data
        self.fs.seek(self.HEADER_DATA_OFFSET)
        h_bytes = self._header_bytes[self.HEADER_DATA_OFFSET:]
        self.fs.write(h_bytes)

        # Flag bytes off as they are now written
        self._header_bytes = None
