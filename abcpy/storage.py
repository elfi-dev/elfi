import os
import pickle
import random
import time
from collections import namedtuple

import numpy as np

from unqlite import UnQLite

class UnQLiteStorage():

    def __init__(self, location=None):
        """
            location (string): path to store the database file
        """
        self.location = location
        self.db = UnQLite(self.location)
        self.collections = dict()

    def add_collection(self, name):
        """
            Add collection to database.

            name (string): collection name
        """
        collection = self.db.collection(name)
        if collection.exists() is False:
            collection.create()
            print("created collection %s" % (name))
            self._commit()
        else:
            print("Collection %s already exist" % (name))
        self.collections[name] = collection

    def _get_collection(self, name):
        """
            Get collection with name from database.

            name (string): collection name
        """
        if name not in self.collections.keys():
            self.add_collection(name)
        return self.collections[name]

    def add_row(self, collection, row):
        """
            Adds row to collection.
            If row does not have 'id' set, a new one will be assigned.
            Returns row id.

            name (string): collection name
            row (dict): row as dictionary
        """
        coll = self._get_collection(collection)
        if "__id" not in row.keys():
            row["__id"] = coll.last_record_id() + 1
        coll.store(row)
        print("stored %s to collection %s" % (row, collection))
        self._commit()
        return row["__id"]

    def get_row_by_id(self, collection, unique_id):
        """
            Returns the row with matching unique_id from collection.

            collection (string): collection name
            unique_id (int): row id
        """
        coll = self._get_collection(collection)
        return coll.fetch(unique_id)

    def _commit(self):
        """
            Commits changes to database, retries few times if database locked.
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
                print("Database locked, waiting %.1fs.." % (delay))
                time.sleep(delay)
                maxtries -= 1
        print("Database error: could not commit")
