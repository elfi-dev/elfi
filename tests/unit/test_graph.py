from elfi.graph import Node


class TestNode:

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
        assert a.descendants == [a, d, e, f, g]
        assert e.descendants == [e, g]
        assert g.descendants == [g]
        assert g.ancestors == [g, e, f, d, c, a, b]
        assert c.ancestors == [c, b]
        assert a.ancestors == [a]
        assert set(d.neighbours) == {e, f, a, c}

    def test_component(self):
        a = Node('a')
        b = Node('b')
        c = Node('c', b)
        d = Node('d', a, c)
        e = Node('e', d)
        f = Node('f', d, c)
        g = Node('g', e, f)
        ext = Node('ext')
        assert set(c.component) == {a, b, c, d, e, f, g}
        assert set(g.component) == {a, b, c, d, e, f, g}
        assert ext.component == [ext]

    def test_consistent_order(self):
        def assert_order(node, attr):
            prev = getattr(node, attr)
            assert isinstance(prev, list)
            for i in range(10):
                cur = getattr(node, attr)
                assert prev == cur
                prev = cur
        a = Node('a')
        b = Node('b')
        c = Node('c', b)
        d = Node('d', a, c)
        e = Node('e', d)
        f = Node('f', d, c)
        g = Node('g', e, f)
        assert_order(g, "component")
        assert_order(g, "ancestors")
        assert_order(g, "parents")
        assert_order(a, "children")
        assert_order(a, "descendants")

    def test_graph_theory_concepts(self):
        root = Node('root')
        # Separate branch
        a1 = Node('a1', root)
        a2 = Node('a2', a1)
        # Alternative root
        root2 = Node('root2')
        # Make a branches that join
        b1 = Node('b1', root)
        c1 = Node('c1', root2)
        bc2 = Node('bc2', b1, c1)

        assert set(a2.component) == {root, root2, a1, a2, b1, c1, bc2}
        assert a2.ancestors == [a2, a1, root]
        assert root2.descendants == [root2, c1, bc2]

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

    def test_none_parent(self):
        a = Node('a', None)
        assert a.parents == []
