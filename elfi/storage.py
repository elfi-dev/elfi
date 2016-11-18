import logging
import os
import pickle
import random
import time
import json
from collections import namedtuple
import io
import numpy as np

from elfi.core import ElfiStore, LocalElfiStore
from elfi.core import get_key_slice, get_key_name
from elfi.utils import to_slice

from unqlite import UnQLite

logger = logging.getLogger(__name__)

def _serialize_numpy(data):
    """For simple numpy arrays.

    Examples
    --------
    >>> ar = np.array([[1], [2]])
    >>> _serialize_numpy(ar)
    '[[1], [2]]'
    """
    l = data.tolist()
    serialized = _serialize_json(l)
    return serialized

def _deserialize_numpy(serialized):
    """For serialized simple numpy arrays.

    Examples
    --------
    >>> ser = "[[1], [2]]"
    >>> _deserialize_numpy(ser)
    array([[1],
           [2]])
    """
    l = _deserialize_json(serialized)
    data = np.array(l)
    return data

def _serialize_json(data):
    """For json-serializable objects.

    Examples
    --------
    >>> ar = [[1], [2]]
    >>> _serialize_json(ar)
    '[[1], [2]]'
    """
    return json.dumps(data)

def _deserialize_json(serialized):
    """For json-serialized data.

    Examples
    --------
    >>> ser = "[[1], [2]]"
    >>> _deserialize_json(ser)
    [[1], [2]]
    """
    return json.loads(serialized)

def _serialize_pickle(data):
    """For pickle-serializable objects.

    Examples
    --------
    >>> ar = [[1], [2]]
    >>> s = _serialize_pickle(ar)
    >>> _deserialize_pickle(s)
    [[1], [2]]
    """
    return pickle.dumps(data)

def _deserialize_pickle(serialized):
    """For pickle-serialized data.
    """
    return pickle.loads(serialized)



class NameIndexDataInterface():
    """An interface for storage objects that allow data to
    be stored based on a name and index or slice.

    'set(name, sl, data)' should imply 'data == get(name, sl)'
    """

    def get(self, name, sl):
        """Return data matching slice as np.array.

        Parameters
        ----------
        name : string
        sl : integer or slice

        Returns
        -------
        Array type with found items matching slice.
        """
        raise NotImplementedError("Subclass implements")

    def set(self, name, sl, data):
        """Store data to indices matching slice.

        Parameters
        ----------
        name : string
        sl : integer or slice
        data : array type with length matching slice
            Contents should be serializable.
        """
        raise NotImplementedError("Subclass implements")


class SerializedStoreInterface(LocalElfiStore):
    """Interface for stores that serialize data.

    ser_type : string (optional)
        if "numpy" uses 'elfi.storage._(de)serialize_numpy'.
        if "json" uses 'elfi.storage._(de)serialize_json'.
        if "pickle" uses 'elfi.storage._(de)serialize_pickle'.
    serizalizer : function(data) -> string (optional)
        Overrides ser_type.
    deserializer : function(string) -> data (optional)
        Overrides ser_type.
    """
    def __init__(self, *args, ser_type=None, serializer=None, deserializer=None, **kwargs):
        self.serialize = None
        self.deserialize = None
        choices = {
            "numpy": (_serialize_numpy, _deserialize_numpy),
            "json": (_serialize_json, _deserialize_json),
            "pickle": (_serialize_pickle, _deserialize_pickle),
            }
        if ser_type in choices.keys():
            self.serialize = serializer or choices[ser_type][0]
            self.deserialize = deserializer or choices[ser_type][1]
        elif ser_type is not None:
            raise ValueError("Unknown serialization type '{}'.".format(ser_type))
        if serializer is not None:
            self.serialize = serializer
        if deserializer is not None:
            self.deserialize = deserializer
        if self.serialize is None:
            raise ValueError("Must define serializer.")
        if self.deserialize is None:
            raise ValueError("Must define deserializer.")
        super(SerializedStoreInterface, self).__init__(*args, **kwargs)


class DictListStore(NameIndexDataInterface, LocalElfiStore):
    """Python dictionary of lists based storage.

    Stores data for each node as:
        "name" : [data_0, ..., data_n]

    Parameters
    ----------
    local_store : dict or None
        If None will create new dict.
    batch_size : int
        How much more space allocate to list when out of space
    """
    def __init__(self, local_store=None, batch_size=100):
        if isinstance(local_store, dict):
            self.store = local_store
        else:
            self.store = {}
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        super(DictListStore, self).__init__()

    def _read_data(self, name, sl):
        """Operation for reading from storage object.

        Parameters
        ----------
        name : string
        sl : slice

        Returns
        -------
        Values matching slice as np.ndarray
        """
        sl = to_slice(sl)
        if name not in self.store.keys():
            return []
        return np.array(self.store[name][sl])

    get = _read_data

    def _write(self, key, output_result):
        """Operation for writing to storage object.

        Parameters
        ----------
        key : dask key
        output_result : dict with keys:
            "data" : np.ndarray
                At least 2D numpy array.
        """
        sl = get_key_slice(key)
        name = get_key_name(key)
        self.set(name, sl, output_result["data"])

    def set(self, name, sl, data):
        if type(sl) == int:
            sl = to_slice(sl)
        if name not in self.store.keys():
            self.store[name] = []
        l = self.store[name]
        while len(l) < sl.stop:
            l.extend([None] * self.batch_size)
        l[sl] = data

    def _reset(self, name):
        """Operation for resetting storage object (optional).
        """
        if name in self.store.keys():
            del self.store[name]


class UnQLiteStore(SerializedStoreInterface, NameIndexDataInterface, LocalElfiStore):
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
    ser_type : string (optional)
    serizalizer : function(data) -> string (optional)
    deserializer : function(string) -> data (optional)
    """
    def __init__(self, local_store=None, ser_type="numpy", serializer=None, deserializer=None):
        if isinstance(local_store, UnQLiteDatabase):
            self.db = local_store
        else:
            self.db = UnQLiteDatabase(local_store)
        super(UnQLiteStore, self).__init__(ser_type=ser_type,
                                           serializer=serializer,
                                           deserializer=deserializer)

    def _read_data(self, name, sl):
        """Operation for reading from storage object.

        Parameters
        ----------
        name : string
        sl : slice

        Returns
        -------
        Values matching slice as np.ndarray
        """
        sl = to_slice(sl)
        filt = lambda row : sl.start <= row["idx"] < sl.stop
        rows = self.db.filter_rows(name, filt)
        try:
            ret = np.array([self.deserialize(row["data"]) for row in rows])
        except Exception as e:
            logger.critical("Could not deserialize data!")
            logger.critical("Error: {}".format(e))
            raise
        return ret

    get = _read_data

    def _write(self, key, output_result):
        """Operation for writing to storage object.

        Parameters
        ----------
        key : dask key
        output_result : dict with keys:
            "data" : np.ndarray
                At least 2D numpy array.
        """
        sl = get_key_slice(key)
        name = get_key_name(key)
        self.set(name, sl, output_result["data"])

    def set(self, name, sl, data):
        if type(sl) == int:
            sl = to_slice(sl)
        i = sl.start
        j = 0
        rows = []
        for i in range(sl.start, sl.stop):
            try:
                ser = self.serialize(data[j])
            except Exception as e:
                logger.critical("Could not serialize data!")
                logger.critical("Error: {}".format(e))
                raise
            rows.append({"idx" : i, "data" : ser})
            j += 1
        self.db.add_rows(name, rows)

    def _reset(self, name):
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
