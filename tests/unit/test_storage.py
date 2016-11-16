import numpy as np
import timeit

import elfi
from elfi.storage import _serialize, _deserialize
from elfi.storage import UnQLiteStore, UnQLiteDatabase

from test_core_persistence import Test_persistence

def test_serialization():
    a = np.array([[1], [2]])
    ser = _serialize(a)
    des = _deserialize(ser)
    np.testing.assert_array_equal(a, des)

def test_database_read_write():
    db = UnQLiteDatabase()
    data1 = {"idx": 1, "A": 2}
    db.add_row("coll", data1)
    rows = db.filter_rows("coll", lambda r: r["idx"] == 1)
    assert type(rows) == list
    assert len(rows) == 1
    data2 = rows[0]
    assert data2["idx"] == data1["idx"]
    assert data2["A"] == data1["A"]


class Test_unqlite_persistence(Test_persistence):

    def test_unqlite_cache(self):
        db = UnQLiteDatabase()
        local_store = UnQLiteStore(db)
        self.run_local_object_cache_test(local_store)
