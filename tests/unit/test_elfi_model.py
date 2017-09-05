import os

import numpy as np
import pytest

import elfi
import elfi.model.elfi_model as em
from elfi.examples import ma2 as ema2


@pytest.mark.usefixtures('with_all_clients')
def test_generate(ma2):
    n_gen = 10

    d = ma2.get_reference('d')
    res = d.generate(n_gen)

    assert res.shape[0] == n_gen
    assert res.ndim == 1


@pytest.mark.usefixtures('with_all_clients')
def test_observed():
    true_params = [.6, .2]
    m = ema2.get_model(100, true_params=true_params)
    y = m.observed['MA2']
    S1 = m.get_reference('S1')
    S2 = m.get_reference('S2')

    S1_observed = ema2.autocov(y)
    S2_observed = ema2.autocov(y, 2)

    assert np.array_equal(S1.observed, S1_observed)
    assert np.array_equal(S2.observed, S2_observed)


def euclidean_discrepancy(*simulated, observed):
    """Euclidean discrepancy between data.

    Parameters
    ----------
    *simulated
        simulated summaries
    observed : tuple of 1d or 2d np.arrays of length n

    Returns
    -------
    d : np.array of size (n,)
    """
    d = np.linalg.norm(np.column_stack(simulated) - np.column_stack(observed), ord=2, axis=1)
    return d


class TestElfiModel:
    def test_remove_node(self, ma2):
        ma2.remove_node('MA2')

        assert not ma2.has_node('MA2')
        assert ma2.has_node('t2')

        parents = ma2.get_parents('t2')
        # This needs to have at least 2 parents so that the test below makes sense
        assert len(parents) > 1

        ma2.remove_node('t2')
        for p in parents:
            if p[0] == '_':
                assert not ma2.has_node(p)
            else:
                assert ma2.has_node(p)

        assert 'MA2' not in ma2.observed

    def test_save_load(self, ma2):
        name = ma2.name
        ma2.save()
        ma2 = elfi.load_model(name)
        os.remove(name + '.pkl')

        # Same with a prefix
        prefix = 'models_dir'
        ma2.save(prefix)
        ma2 = elfi.load_model(name, prefix)
        os.remove(os.path.join(prefix, name + '.pkl'))
        os.removedirs(prefix)


class TestNodeReference:
    def test_name_argument(self):
        # This is important because it is used when passing NodeReferences as
        # InferenceMethod arguments
        em.set_default_model()
        ref = em.NodeReference(name='test')
        assert str(ref) == 'test'

    def test_name_determination(self):
        em.set_default_model()
        node = em.NodeReference()
        assert node.name == 'node'

        # Works without spaces
        node2 = em.NodeReference()
        assert node2.name == 'node2'

        # Does not give the same name
        node = em.NodeReference()
        assert node.name != 'node'

        # Works with sub classes
        pri = em.Prior('uniform')
        assert pri.name == 'pri'

        # Assigns random names when the name isn't self explanatory
        nodes = []
        for i in range(5):
            nodes.append(em.NodeReference())

        for i in range(1, 5):
            assert nodes[i - 1].name != nodes[i].name

    def test_positional_parents(self, ma2):
        true_positional_parents = ['S1', 'S2']
        # This tests that the order of the list is deterministic (no randomness resulting
        # from direct hash to list conversion)
        for i in range(100):
            assert [p.name for p in ma2['d'].parents] == true_positional_parents

    def test_become(self, ma2):
        state = np.random.get_state()
        dists = ma2.generate(100, 'd')['d']
        nodes = ma2.nodes
        np.random.set_state(state)

        ma2['d'].become(em.Discrepancy(euclidean_discrepancy, ma2['S1'], ma2['S2']))
        dists2 = ma2.generate(100, 'd')['d']
        nodes2 = ma2.nodes
        np.random.set_state(state)

        assert np.array_equal(dists, dists2)

        # Check that there are the same nodes in the graph
        assert set(nodes) == set(nodes2)

    def test_become_with_priors(self, ma2):
        parameters = ma2.parameter_names.copy()
        parent_names = ma2.get_parents('t1')

        ma2['t1'].become(elfi.Prior('uniform', 0, model=ma2))

        # Test that parameters are preserved
        assert parameters == ma2.parameter_names

        # Test that hidden nodes are removed
        for name in parent_names:
            assert not ma2.has_node(name)

        # Test that inference still works
        r = elfi.Rejection(ma2, 'd')
        r.sample(10)

    def test_become_with_simulators(self, ma2):
        y_obs = np.zeros(100)
        new_sim = elfi.Simulator(ema2.MA2, ma2['t1'], ma2['t2'], observed=y_obs)
        ma2['MA2'].become(new_sim)

        # Test that observed data is changed
        assert np.array_equal(ma2.observed['MA2'], y_obs)

        # Test that inference still works
        r = elfi.Rejection(ma2, 'd')
        r.sample(10)
