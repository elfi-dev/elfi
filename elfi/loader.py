"""Loading makes precomputed data accessible to nodes."""

import numpy as np

from elfi.utils import get_sub_seed, observed_name


class Loader:
    """Base class for Loaders."""

    @classmethod
    def load(cls, context, compiled_net, batch_index):
        """Load precomputed data into nodes of `compiled_net`.

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
        raise NotImplementedError


class ObservedLoader(Loader):  # noqa: D101
    @classmethod
    def load(cls, context, compiled_net, batch_index):
        """Add the observed data to the `compiled_net`.

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
        observed = compiled_net.graph['observed']

        for name, obs in observed.items():
            obs_name = observed_name(name)
            if not compiled_net.has_node(obs_name):
                continue
            compiled_net.nodes[obs_name].update(dict(output=obs))
            del compiled_net.nodes[obs_name]['operation']

        del compiled_net.graph['observed']
        return compiled_net


class AdditionalNodesLoader(Loader):  # noqa: D101
    @classmethod
    def load(cls, context, compiled_net, batch_index):
        """Add runtime information to instruction nodes.

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
        meta_dict = {
            'batch_index': batch_index,
            'submission_index': context.num_submissions,
            'master_seed': context.seed,
            'model_name': compiled_net.graph['name']
        }

        details = dict(_batch_size=context.batch_size, _meta=meta_dict)

        for node, v in details.items():
            if node in compiled_net:
                compiled_net.nodes[node]['output'] = v
        return compiled_net


class PoolLoader(Loader):  # noqa: D101
    @classmethod
    def load(cls, context, compiled_net, batch_index):
        """Add data from the pools in `context`.

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
        if context.pool is None:
            return compiled_net

        batch = context.pool.get_batch(batch_index)

        for node in context.pool.stores:
            if not compiled_net.has_node(node):
                continue
            elif node in batch:
                compiled_net.nodes[node]['output'] = batch[node]
                compiled_net.nodes[node].pop('operation', None)
            elif node not in compiled_net.graph['outputs']:
                # We are missing this item from the batch so add the output to the
                # requested outputs so that it can be stored when the results arrive
                compiled_net.graph['outputs'].add(node)

        return compiled_net


# We use a getter function so that the local process np.random doesn't get
# copied to the loaded_net.
def get_np_random():
    """Get RandomState."""
    return np.random.mtrand._rand


class RandomStateLoader(Loader):  # noqa: D101
    @classmethod
    def load(cls, context, compiled_net, batch_index):
        """Add an instance of random state to the corresponding node.

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
        key = 'output'
        seed = context.seed

        if seed == 'global':
            # Get the random_state of the respective worker by delaying the evaluation
            random_state = get_np_random
            key = 'operation'
        elif isinstance(seed, (int, np.int32, np.uint32)):
            # TODO: In the future, we could use https://pypi.python.org/pypi/randomstate to enable
            # jumps?
            cache = context.caches.get('sub_seed', None)
            sub_seed = get_sub_seed(seed, batch_index, cache=cache)
            random_state = np.random.RandomState(sub_seed)
        else:
            raise ValueError("Seed of type {} is not supported".format(seed))

        # Assign the random state or its acquirer function to the corresponding node
        node_name = '_random_state'
        if compiled_net.has_node(node_name):
            compiled_net.nodes[node_name][key] = random_state

        return compiled_net
