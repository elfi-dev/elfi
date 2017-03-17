import os
import shutil
import io

import numpy as np
import numpy.lib.format as npformat


class OutputPool:

    def __init__(self, outputs=None, model_name=None, basepath=None):
        self.outputs = set(outputs or [])
        self.stores = {}

        self.model_name = model_name

        self.basepath = basepath or os.path.join(os.path.expanduser('~'), '.elfi')
        os.makedirs(self.basepath, exist_ok=True)

    def __getitem__(self, item):
        return self.stores[item]

    def _add_output_store(self, name):
        raise NotImplementedError


class Store:

    def __init__(self, n_batches, batch_size, mask=None, batch_offset=0):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.batches_mask = mask or np.zeros(n_batches, dtype=bool)
        self.batch_offset = batch_offset

    def has_batch(self, index):
        if self.batch_offset > index or index >= self.batch_offset + self.n_batches:
            return False
        return self.batches_mask[index]


class NpyFileAppender:
    """Appends new data to the end of a .npy file. Existing data region is guaranteed to
    stay unmodified.

    Notes
    -----
    - Currently supports only binary files.
    - Currently supports only .npy version 2.0
    - See numpy.lib.npformat documentation for a description of the .npy npformat """

    MAX_SHAPE_LEN = 2**64

    # Version 2.0 header prefix length
    TO_HEADER_DATA = 12
    TO_HEADER_SIZE = 8

    def __init__(self, filename, mode='a'):
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

        self.filename = filename

        if mode == 'a' and os.path.exists(self.filename):
            self.fs = open(self.filename, 'r+b')
            self._read_header()
        else:
            self.fs = open(self.filename, 'w+b')

    def _read_header(self):
        self.fs.seek(self.TO_HEADER_SIZE)
        self.shape, self.fortran_order, self.dtype = npformat.read_array_header_2_0(self.fs)
        self.header_length = self.fs.tell()

    def append(self, array):
        if self.header_length is None:
            self._init_file(array)
            return

        # Append new data
        self.fs.seek(0, 2)
        self.fs.write(array.tobytes('C'))
        self.shape = (self.shape[0] + len(array),) + self.shape[1:]
        self._write_header_data()

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
        fs.write(h_bytes.read(self.TO_HEADER_DATA))

        # Write correct header data
        self._write_header_data()
        pos = self.fs.tell()

        # Pad
        fill_len = self.header_length - pos
        if fill_len:
            fs.write(b'\x20' * fill_len)

        # Write data
        fs.write(array.tobytes('C'))

    def close(self):
        if not self.fs.closed:
            self.fs.flush()
            self.fs.close()

    def flush(self):
        self.fs.flush()

    def __del__(self):
        self.close()

    def _write_header_data(self):
        # Make header data
        d = {
            'shape': self.shape,
            'fortran_order': self.fortran_order,
            'descr': npformat.dtype_to_descr(self.dtype)
        }

        h_bytes = io.BytesIO()
        npformat.write_array_header_2_0(h_bytes, d)
        h_pos = h_bytes.tell()

        if h_pos > self.header_length:
            raise OverflowError("File {} cannot be appended. The header is too short.".
                                format(self.filename))

        # Rewrite header data
        self.fs.seek(self.TO_HEADER_DATA)
        h_bytes.seek(self.TO_HEADER_DATA)
        self.fs.write(h_bytes.read(-1))


class FileStore(Store):
    """Wrapper around numpy.memmap"""

    def __init__(self, path):
        mode = 'r+' if os.path.exists(path) else 'w+'
        self.file = np.load(path, mmap_mode=mode)

    def add_batch(self, batch_output, batch_index):
        outputs = self.outputs or batch_output.keys()
        save_outputs = {k:batch_output[k] for k in outputs}
        filepath = self.filepath(batch_index)
        np.savez(filepath, **save_outputs)

    def has_batch(self, index):
        return os.path.exists(self.filepath(index))

    def read_batch(self, batch_index):
        filepath = self.filepath(batch_index)
        return np.load(filepath)

    @property
    def path(self):
        if not self.model_name:
            raise ValueError('Model name is not specified')
        return os.path.join(self.basepath, self.model_name)

    def filepath(self, batch_index):
        filename = 'batch_' + str(batch_index) + '.npz'
        os.makedirs(self.path, exist_ok=True)
        return os.path.join(self.path, filename)

    def destroy(self):
        shutil.rmtree(self.path)

    def test(self):
        pass
        np.load()
        np.memmap
        np.open_mem