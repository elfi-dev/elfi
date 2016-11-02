import numpy as np
from functools import partial

from elfi.core import simulator_operation, Node

class MockSimulator():

    def __init__(self, rets):
        self.n_calls = 0
        self.args = list()
        self.kwargs = list()
        self.rets = rets

    def __call__(self, *args, **kwargs):
        self.args.append(args)
        self.kwargs.append(kwargs)
        kwargs["prng"].rand()
        ret = self.rets[self.n_calls]
        self.n_calls += 1
        return ret


class Test_simulator_operation():

    def test_vectorized(self):
        ret1 = np.atleast_2d([[5], [6]])
        mock = MockSimulator([ret1])
        prng = np.random.RandomState(1234)
        input_dict = {
                "n": 2,
                "data": [np.atleast_2d([[1], [2]]),
                         np.atleast_2d([[3], [4]])],
                "random_state": prng.get_state()
                }
        output_dict = simulator_operation(mock, True, input_dict)
        prng.rand()
        print(output_dict)
        assert mock.n_calls == 1
        assert output_dict["n"] == 2
        assert np.array_equal(output_dict["data"], ret1)
        new_state = prng.get_state()
        assert output_dict["random_state"][0] == new_state[0]
        assert np.array_equal(output_dict["random_state"][1], new_state[1])
        assert output_dict["random_state"][2] == new_state[2]
        assert output_dict["random_state"][3] == new_state[3]
        assert output_dict["random_state"][4] == new_state[4]

    def test_sequential(self):
        ret1 = np.array([5])
        ret2 = np.array([6])
        mock = MockSimulator([ret1, ret2])
        prng = np.random.RandomState(1234)
        input_dict = {
                "n": 2,
                "data": [np.atleast_2d([[1], [2]]),
                         np.atleast_2d([[3], [4]])],
                "random_state": prng.get_state()
                }
        output_dict = simulator_operation(mock, False, input_dict)
        prng.rand()
        prng.rand()
        print(output_dict)
        assert mock.n_calls == 2
        assert output_dict["n"] == 2
        assert np.array_equal(output_dict["data"], np.vstack((ret1, ret2)))
        new_state = prng.get_state()
        assert output_dict["random_state"][0] == new_state[0]
        assert np.array_equal(output_dict["random_state"][1], new_state[1])
        assert output_dict["random_state"][2] == new_state[2]
        assert output_dict["random_state"][3] == new_state[3]
        assert output_dict["random_state"][4] == new_state[4]


class Test_Node():
    def test_construction1(self):
        a = Node('a')
        assert a.name == 'a'
        assert a.parents == []
        assert a.children == []
        assert a.is_root()
        assert a.is_leaf()

    def test_construction(self):
        a = Node('a')
        b = Node('b')
        c = Node('c', b)
        d = Node('d', a, c)
        e = Node('e', d)
        f = Node('f', d, c)
        g = Node('g', e, f)
        assert b.is_root()
        assert c.parents == [b]
        assert d.parents == [a, c]
        assert g.parents == [e, f]
        assert b.children == [c]
        assert c.children == [d, f]
        assert g.is_leaf()

    def test_construction_add(self):
        b = Node('b')
        c = Node('c')
        d = Node('d')
        c.add_parent(b)
        assert c.parents == [b]
        assert b.children == [c]
        b._add_child(d)
        assert b.children == [c, d]
        assert d.parents == []  # _add_child doesn't call add_parent

    def test_construction_unique(self):
        b = Node('b')
        c = Node('c', b)
        c.add_parent(b)
        assert c.parents == [b]
        assert b.children == [c]

    def test_construction_add_index(self):
        b = Node('b')
        c = Node('c', b)
        d = Node('d')
        e = Node('e')
        c.add_parent(d, index=0, index_child=0)
        assert c.parents == [d, b]
        assert b.children == [c]
        assert d.children == [c]
        b._add_child(e, index=0)
        assert b.children == [e, c]
        assert e.parents == []  # _add_child doesn't call add_parent

    def test_construction_index_out_of_bounds(self):
        b = Node('b')
        c = Node('c')
        try:
            c.add_parent(b, index=13)
            b._add_child(c, index=13)
            assert False
        except ValueError:
            assert True

    def test_construction_acyclic(self):
        a = Node('a')
        b = Node('b', a)
        c = Node('c', b)
        try:
            a.add_parent(c)
            assert False
        except ValueError:
            assert True

    def test_family_tree(self):
        a = Node('a')
        b = Node('b')
        c = Node('c', b)
        d = Node('d', a, c)
        e = Node('e', d)
        f = Node('f', d, c)
        g = Node('g', e, f)
        assert a.descendants == [d, e, f, g]
        assert e.descendants == [g]
        assert g.descendants == []
        assert g.ancestors == [e, f, d, a, c, b]
        assert c.ancestors == [b]
        assert a.ancestors == []
        assert d.neighbours == [e, f, a, c]

    def test_component(self):
        a = Node('a')
        b = Node('b')
        c = Node('c', b)
        d = Node('d', a, c)
        e = Node('e', d)
        f = Node('f', d, c)
        g = Node('g', e, f)
        assert c.component == [c, b, d, f, e, g]
        assert g.component == [g, e, f, d, a, c, b]

    def test_remove_parent(self):
        a = Node('a')
        b = Node('b', a)
        c = Node('c', b)
        c.remove_parent(0)
        assert c.parents == []
        assert b.children == []
        b.remove_parent(a)
        assert b.parents == []
        assert a.children == []

    def test_remove(self):
        a = Node('a')
        b = Node('b')
        c = Node('c', a, b)
        d = Node('d', c)
        e = Node('e', c)
        c.remove(keep_parents=False, keep_children=False)
        assert c.children == []
        assert c.parents == []
        assert b.children == []
        assert e.parents == []

    def test_change_to(self):
        a = Node('a')
        b = Node('b')
        c = Node('c', a, b)
        d = Node('d', c)
        e = Node('e', c)
        f = Node('f')
        c_new = Node('c_new', a, f)
        c = c.change_to(c_new, transfer_parents=False, transfer_children=True)
        assert c.parents == [a, f]
        assert c.children == [d, e]
        c_new2 = Node('c_new2', a)
        c = c.change_to(c_new2, transfer_parents=True, transfer_children=False)
        assert c.parents == [a, f]
        assert c.children == []

