import ipyparallel
import pytest

import elfi


@pytest.mark.usefixtures('with_all_clients')
def test_meta_param(ma2):
    sim = ma2.get_reference('MA2')

    # Test that it is passed
    try:
        # Add to state
        sim.uses_meta = True
        sim.generate()
        assert False, "Should raise an error"
    except TypeError:
        assert True
    except ipyparallel.error.RemoteError:
        assert True


# TODO: add ipyparallel, maybe use dill or cloudpickle
# client.ipp_client[:].use_dill() or .use_cloudpickle()
def test_batch_index_value(ma2):
    def bi(meta):
        return meta['batch_index']

    # Test the correct batch_index value
    op = elfi.Operation(bi, model=ma2, name='op')
    op.uses_meta = True
    client = elfi.get_client()
    c = elfi.ComputationContext()
    compiled_net = client.compile(ma2.source_net, ma2.nodes)
    loaded_net = client.load_data(compiled_net, c, batch_index=3)
    res = client.compute(loaded_net)

    assert res['op'] == 3


def test_reduce_compiler(ma2, client):
    compiled_net = client.compile(ma2.source_net)
    assert compiled_net.has_node('S1')

    compiled_net2 = client.compile(ma2.source_net, ['MA2'])
    assert not compiled_net2.has_node('S1')
