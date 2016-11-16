import logging
import os
import pickle
import random
import time
import json
from collections import namedtuple
import io
import numpy as np

from elfi.core import LocalDataStoreInterface
from elfi.utils import to_slice

from unqlite import UnQLite

logger = logging.getLogger(__name__)

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


class UnQLiteStore(LocalDataStoreInterface):
    """UnQLite database based storage.

    Writes to table 'str(self._output_name)' so same underlying
    database can be used for multiple nodes.

    Parameters
    ----------
    local_store : UnQLiteDatabase object or None
        If None will create in-memory database.
    serizalizer : function(data) -> string (optional)
        If not given, uses 'elfi.storage._serialize'.
    deserializer : function(string) -> data (optional)
        If not given, uses 'elfi.storage._deserialize'.
    """
    def __init__(self, local_store=None, serializer=None, deserializer=None):
        if not isinstance(local_store, UnQLiteDatabase):
            raise ValueError("Only compatible with UnQLiteDatabase.")
        self.serialize = serializer or _serialize
        self.deserialize = deserializer or _deserialize
        super(UnQLiteStore, self).__init__(local_store)

    def __getitem__(self, sl):
        """Operation for reading from storage object.

        Returns
        -------
        Values matching slice as np.ndarray
        """
        if self._output_name is None:
            return []
        sl = to_slice(sl)
        i = sl.start
        filt = lambda row : sl.start <= row["idx"] < sl.stop
        rows = self._local_store.filter_rows(self._output_name, filt)
        ret = np.array([self.deserialize(row["data"]) for row in rows])
        return ret

    def __setitem__(self, sl, data):
        """Operation for writing to storage object.

        data : np.ndarray
            At least 2D numpy array.
        """
        if self._output_name is None:
            raise AssertionError("Assumed self._output_name would have been set by write().")
        sl = to_slice(sl)
        i = sl.start
        j = 0
        rows = []
        for i in range(sl.start, sl.stop):
            rows.append({"idx" : i, "data" : self.serialize(data[j])})
            j += 1
        self._local_store.add_rows(self._output_name, rows)

    def _reset_op(self):
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
            self.db = UnQLite(self.location)
        else:
            # in-memory database
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

