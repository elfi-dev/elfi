import numpy as np

from elfi.utils import observed_name, get_sub_seed, is_array


class Loader:
    """
    Loads precomputed values to the compiled network
    """
    @classmethod
    def load(cls, context, compiled_net, batch_index):
        """Load data into nodes of compiled_net

        Parameters
        ----------
        context : ComputationContext
        compiled_net : nx.DiGraph
        batch_index : int

        Returns
        -------
        net : nx.DiGraph
            Loaded net, which is the `compiled_net` that has been loaded with data that
            can depend on the batch_index.
        """


class ObservedLoader(Loader):
    """
    Add the observed data to the compiled net
    """

    @classmethod
    def load(cls, context, compiled_net, batch_index):
        observed = compiled_net.graph['observed']

        for name, obs in observed.items():
            obs_name = observed_name(name)
            if not compiled_net.has_node(obs_name):
                continue
            compiled_net.node[obs_name] = dict(output=obs)

        del compiled_net.graph['observed']
        return compiled_net


class AdditionalNodesLoader(Loader):
    @classmethod
    def load(cls, context, compiled_net, batch_index):
        meta_dict = {'batch_index': batch_index,
                     'submission_index': context.num_submissions,
                     'master_seed': context.seed,
                     'model_name': compiled_net.graph['name']
                     }

        details = dict(_batch_size=context.batch_size,
                       _meta=meta_dict)

        for node, v in details.items():
            if compiled_net.has_node(node):
                compiled_net.node[node]['output'] = v

        return compiled_net


class PoolLoader(Loader):

    @classmethod
    def load(cls, context, compiled_net, batch_index):
        if context.pool is None:
            return compiled_net

        batch = context.pool.get_batch(batch_index)

        for node in context.pool.stores:
            if not compiled_net.has_node(node):
                continue
            elif node in batch:
                compiled_net.node[node]['output'] = batch[node]
                compiled_net.node[node].pop('operation', None)
            elif node not in compiled_net.graph['outputs']:
                # We are missing this item from the batch so add the output to the
                # requested outputs so that it can be stored when the results arrive
                compiled_net.graph['outputs'].append(node)

        return compiled_net


# We use a getter function so that the local process np.random doesn't get
# copied to the loaded_net.
def get_np_random():
    return np.random.mtrand._rand


class RandomStateLoader(Loader):
    """
    Add random state instance for the node
    """

    @classmethod
    def load(cls, context, compiled_net, batch_index):
        key = 'output'
        seed = context.seed
        if seed is 'global':
            # Get the random_state of the respective worker by delaying the evaluation
            random_state = get_np_random
            key = 'operation'
        elif isinstance(seed, (int, np.int32, np.uint32)):
            random_state = np.random.RandomState(context.seed)
        else:
            raise ValueError("Seed of type {} is not supported".format(seed))

        # Jump (or scramble) the state based on batch_index to create parallel separate
        # pseudo random sequences
        if seed is not 'global':
            # TODO: In the future, we could use https://pypi.python.org/pypi/randomstate to enable jumps?
            random_state = np.random.RandomState(get_sub_seed(random_state, batch_index))

        _random_node = '_random_state'
        if compiled_net.has_node(_random_node):
            compiled_net.node[_random_node][key] = random_state

        return compiled_net
