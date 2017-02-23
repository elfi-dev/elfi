from elfi.compiler import ObservedCompiler
from elfi.utils import splen


class Loader:
    """
    Loads precomputed values to the compiled network
    """
    @classmethod
    def load(cls, model, output_net, span):
        """

        Parameters
        ----------
        model : ElfiModel
        output_net : nx.DiGraph
        span : tuple

        Returns
        -------

        """


class ObservedLoader(Loader):
    """
    Add observed data to computation graph
    """

    @classmethod
    def load(cls, context, output_net, span):
        for name, v in context.observed.items():
            obs_name = ObservedCompiler.obs_name(name)
            if not output_net.has_node(obs_name):
                continue
            output_net.node[obs_name] = dict(output=v)

        return output_net


class BatchSizeLoader(Loader):
    """
    Add observed data to computation graph
    """

    @classmethod
    def load(cls, context, output_net, span):
        _name = '_batch_size'
        if output_net.has_node(_name):
            output_net.node[_name]['output'] = splen(span)

        return output_net


class RandomStateLoader(Loader):
    """
    Add random state instance for the node
    """

    @classmethod
    def load(cls, context, output_net, span):
        _name = '_{}_random_state'
        # If a need arises to reduce iterations over all nodes, we can rename these as
        # e.g. random_state_0, random_state_1, ...
        for node, d in output_net.nodes_iter(data=True):
            random_node = _name.format(node)
            if output_net.has_node(random_node):
                output_net.node[random_node]['output'] = None

        return output_net
