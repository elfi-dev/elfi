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
    def load(cls, model, output_net, span):
        for name, v in model.observed.items():
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
    def load(cls, model, output_net, span):
        _name = '_batch_size'
        if output_net.has_node(_name):
            output_net.node[_name]['output'] = splen(span)

        return output_net