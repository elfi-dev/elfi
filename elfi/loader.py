import numpy as np

from elfi.utils import splen, observed_name


class Loader:
    """
    Loads precomputed values to the compiled network
    """
    @classmethod
    def load(cls, context, output_net, batch_index):
        """

        Parameters
        ----------
        context : ComputationContext
        output_net : nx.DiGraph
        batch_index : int

        Returns
        -------

        """


class ObservedLoader(Loader):
    """
    Add observed data to computation graph
    """

    @classmethod
    def load(cls, context, output_net, batch_index):
        for name, v in context.observed.items():
            obs_name = observed_name(name)
            if not output_net.has_node(obs_name):
                continue
            output_net.node[obs_name] = dict(output=v)

        return output_net


class BatchSizeLoader(Loader):
    """
    Add observed data to computation graph
    """

    @classmethod
    def load(cls, context, output_net, batch_index):
        _name = '_batch_size'
        if output_net.has_node(_name):
            output_net.node[_name]['output'] = context.batch_size

        return output_net


class RandomStateLoader(Loader):
    """
    Add random state instance for the node
    """

    @classmethod
    def load(cls, context, output_net, batch_index):
        if context.seed is None:
            random_state = None
        else:
            random_state = np.random.RandomState(context.seed)

        _random_node = '_random_state'
        if output_net.has_node(_random_node):
            output_net.node[_random_node]['output'] = random_state

        return output_net
