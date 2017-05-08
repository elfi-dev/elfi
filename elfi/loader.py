import numpy as np

from elfi.utils import observed_name, get_sub_seed


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


class BatchMetaLoader(Loader):
    """Adds values to _batch_size and _batch_index nodes if they are present.
    """

    @classmethod
    def load(cls, context, output_net, batch_index):
        details = dict(_batch_size=context.batch_size,
                       _batch_index=batch_index)

        for node, v in details.items():
            if output_net.has_node(node):
                output_net.node[node]['output'] = v

        return output_net


class PoolLoader(Loader):

    @classmethod
    def load(cls, context, output_net, batch_index):
        if context.pool is None:
            return output_net

        batch = context.pool.get_batch(batch_index)

        for node in context.pool.output_stores:
            if not output_net.has_node(node):
                continue
            elif node in batch:
                output_net.node[node]['output'] = batch[node]
                output_net.node[node].pop('operation', None)
            elif node not in output_net.graph['outputs']:
                # We are missing this item from the batch so add the output to the
                # requested outputs so that it can be stored when the results arrive
                output_net.graph['outputs'].append(node)

        return output_net


# We use a getter function so that the local process np.random doesn't get
# copied to the loaded_net.
def get_np_random():
    return np.random.mtrand._rand


class RandomStateLoader(Loader):
    """
    Add random state instance for the node
    """

    @classmethod
    def load(cls, context, output_net, batch_index):
        key = 'output'
        seed = context.seed
        if seed is False:
            # Get the random_state of the respective worker by delaying the evaluation
            random_state = get_np_random
            key = 'operation'
        elif isinstance(seed, (int, np.uint32)):
            random_state = np.random.RandomState(context.seed)
        else:
            raise ValueError("Seed of type {} is not supported".format(seed))

        # Jump (or scramble) the state based on batch_index to create parallel separate
        # pseudo random sequences
        if seed is not False:
            # TODO: In the future, allow use of https://pypi.python.org/pypi/randomstate ?
            random_state = np.random.RandomState(get_sub_seed(random_state, batch_index))

        _random_node = '_random_state'
        if output_net.has_node(_random_node):
            output_net.node[_random_node][key] = random_state

        return output_net
