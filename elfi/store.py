import logging
import os
import random
import time
import json
from collections import defaultdict

import numpy as np
from dask.delayed import delayed
from tornado import gen

from elfi.utils import to_slice, get_key_slice, get_key_name, get_named_item, make_key
import elfi.env as env

from unqlite import UnQLite

logger = logging.getLogger(__name__)


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


"""
Implementations
"""


def _serialize(data):
    """For simple numpy arrays.
    """
    l = data.tolist()
    serialized = json.dumps(l)
    return serialized

def _deserialize(serialized):
    """For simple numpy arrays.
    """
    l = json.loads(serialized)
    data = np.array(l)
    return data


class UnQLiteStore(LocalElfiStore):
    """UnQLite database based storage.

    Stores data for each node to collection with same name,
    so same instance can be used for multiple nodes.

    Each row (dict) in the underlying database will have format:
        "idx": int, matches slice index
        "data": serialized data
        "__id": internal row uuid

    Parameters
    ----------
    local_store : UnQLiteDatabase object or filename or None
        If None will create in-memory database.
    serizalizer : function(data) -> string (optional)
        If not given, uses 'elfi.storage._serialize'.
    deserializer : function(string) -> data (optional)
        If not given, uses 'elfi.storage._deserialize'.
    """
    def __init__(self, local_store=None, serializer=None, deserializer=None):
        if isinstance(local_store, UnQLiteDatabase):
            self.db = local_store
        else:
            self.db = UnQLiteDatabase(local_store)
        self.serialize = serializer or _serialize
        self.deserialize = deserializer or _deserialize
        super(UnQLiteStore, self).__init__()

    def _read_data(self, name, sl):
        """Operation for reading from storage object.

        Returns
        -------
        Values matching slice as np.ndarray
        """
        sl = to_slice(sl)
        filt = lambda row : sl.start <= row["idx"] < sl.stop
        rows = self.db.filter_rows(name, filt)
        ret = np.array([self.deserialize(row["data"]) for row in rows])
        return ret

    get = _read_data

    def _write(self, key, output_result):
        """Operation for writing to storage object.

        data : np.ndarray
            At least 2D numpy array.
        """
        sl = get_key_slice(key)
        name = get_key_name(key)
        i = sl.start
        j = 0
        rows = []
        for i in range(sl.start, sl.stop):
            ser = self.serialize(output_result["data"][j])
            rows.append({"idx" : i, "data" : ser})
            j += 1
        self.db.add_rows(name, rows)

    def _reset(self):
        """Operation for resetting storage object (optional).
        """
        pass


class UnQLiteDatabase():
    """UnQLite database wrapper.

    Parameters
    ----------
    location : string
        Path to store the database file.
        If not given, make in-memory database.
    """

    def __init__(self, location=None):
        self.location = location
        if type(self.location) == str and len(self.location) > 0:
            logger.debug("Connecting to database at {}".format(os.path.abspath(location)))
            self.db = UnQLite(self.location)
        else:
            # in-memory database
            logger.debug("Creating an in-memory database.")
            self.db = UnQLite()
        self.collections = dict()

    def add_collection(self, name):
        """Add collection to database and create it if it doesn't yet exist.

        Parameters
        ----------
        name : string
            Collection name.
        """
        if name in self.collections.keys():
            # assume already exists
            return
        collection = self.db.collection(name)
        if collection.exists() is False:
            # does not exist at all yet
            collection.create()
            logger.debug("({}) Created collection {}".format(self.location, name))
            self._commit()
        self.collections[name] = collection

    def _get_collection(self, name):
        """Get collection with name from database.

        Parameters
        ----------
        name : string
            Collection name.

        Returns
        -------
        Collection
        """
        if name not in self.collections.keys():
            self.add_collection(name)
        return self.collections[name]

    def add_row(self, collection, row):
        self.add_rows(collection, [row])

    def add_rows(self, collection, rows):
        """Adds row to collection.

        Parameters
        ----------
        name : string
            Collection name.
        row : list of dicts
            Rows to store.
        """
        coll = self._get_collection(collection)
        coll.store(rows)
        self._commit()

    def filter_rows(self, collection, filt):
        """Returns the rows matching filter.

        Parameters
        ----------
        collection : string
            Collection name.
        filter : function(row) -> bool
            Filter function that returns True for items to return

        Returns
        -------
        Row : List of matching rows
        """
        coll = self._get_collection(collection)
        return coll.filter(filt)

    def _commit(self):
        """Commits changes to database, retries few times if database locked.
        """
        maxtries = 10
        while True:
            try:
                self.db.commit()
                return
            except:
                if maxtries < 1:
                    raise
                self.db.rollback()
                delay = max(0.5, min(random.expovariate(3.0), 10.0))
                logger.debug("({}) Database locked, waiting {:.1f}s.."
                        .format(self.location, delay))
                time.sleep(delay)
                maxtries -= 1
        logger.warning("({}) Database error: could not commit!".format(self.location))

