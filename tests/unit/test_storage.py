import numpy as np
import os

import elfi
from elfi.storage import _serialize, _deserialize
from elfi.storage import UnQLiteStore, UnQLiteDatabase

from test_core_persistence import Test_persistence

def test_serialization():
    a = np.array([[1], [2]])
    ser = _serialize(a)
    des = _deserialize(ser)
    np.testing.assert_array_equal(a, des)

def database_read_write_test(db):
    data1 = {"idx": 1, "A": 2}
    rows = db.filter_rows("coll", lambda r: r["idx"] == 1)
    assert len(rows) == 0
    db.add_row("coll", data1)
    rows = db.filter_rows("coll", lambda r: r["idx"] == 1)
    assert len(rows) == 1
    data2 = rows[0]
    assert data2["idx"] == data1["idx"]
    assert data2["A"] == data1["A"]
    data3 = [{"idx": 2, "A": 2},
             {"idx": 3, "A": 3}]
    data4 = {"idx": 1, "A": 2}
    db.add_rows("coll", data3)
    db.add_rows("coll2", data4)
    rows = db.filter_rows("coll", lambda r: r["A"] == 2)
    assert len(rows) == 2
    rows = db.filter_rows("coll2", lambda r: r["A"] == 2)
    assert len(rows) == 1

def database_write_test(db):
    data1 = [{"idx": 1, "A": 2},
             {"idx": 2, "A": 3},
             {"idx": 3, "A": 3},
             {"idx": 5, "A": 3, "B": 1}]
    data2 = {"idx": 1, "A": 2}
    db.add_rows("coll", data1)
    db.add_row("coll2", data2)

def database_read_test(db):
    assert len(db.filter_rows("coll", lambda r: r["A"] == 2)) == 1
    assert len(db.filter_rows("coll", lambda r: r["A"] == 3)) == 3
    assert len(db.filter_rows("coll", lambda r: r["B"] == 1)) == 1
    assert len(db.filter_rows("coll2", lambda r: r["A"] == 2)) == 1
    assert len(db.filter_rows("coll2", lambda r: r["A"] == 1)) == 0

def test_in_memeory_database():
    db = UnQLiteDatabase()
    database_read_write_test(db)

def test_file_database():
    fn = "test_db_128376128376213.temp"
    open(fn, "w").close()
    assert os.path.isfile(fn)
    try:
        db = UnQLiteDatabase(fn)
        database_write_test(db)
        database_read_test(db)
        del db
        db2 = UnQLiteDatabase(fn)
        database_read_test(db2)
    finally:
        os.remove(fn)


class Test_unqlite_persistence(Test_persistence):

    def test_unqlite_cache(self):
        local_store = UnQLiteStore()
        self.run_local_object_cache_test(local_store)
