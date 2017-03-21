import numpy as np

from elfi.utils import observed_name


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


# TODO: merge to PoolLoader
class OutputSupplyLoader(Loader):

    @classmethod
    def load(cls, context, output_net, batch_index):
        for node, supply in context.output_supply.items():
            output_net.node[node]['output'] = supply[batch_index][node]
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
            elif node not in output_net.graph['outputs']:
                # Add output so that it can be stored when the results come
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
        seed = context.seed
        if seed is False:
            # Get the random_state of the respective worker by delaying the evaluation
            random_state = get_np_random
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
            output_net.node[_random_node]['output'] = random_state

        return output_net


def get_sub_seed(random_state, sub_seed_index, high=2**32):
    """Returns a sub seed. The returned sub seed is unique for its index, i.e. no
    two indexes can return the same sub_seed. Same random_state will also always
    produce the same sequence.

    Parameters
    ----------
    random_state : np.random.RandomState
    sub_seed_index : int
    high : int
        upper limit for the range of sub seeds (exclusive)

    Returns
    -------
    int
        from interval [0, high - 1]

    Notes
    -----
    There is no guarantee how close the random_states initialized with sub_seeds may end
    up to each other. Better option is to use PRNG:s that have an advance or jump
    functions available.

    """

    if sub_seed_index >= high:
        raise ValueError("Sub seed index {} is out of range".format(sub_seed_index))

    n_unique = 0
    n_unique_required = sub_seed_index + 1
    sub_seeds = None
    seen = set()
    while n_unique != n_unique_required:
        n_draws = n_unique_required - n_unique
        sub_seeds = random_state.randint(high, size=n_draws)
        seen.update(sub_seeds)
        n_unique = len(seen)

    return sub_seeds[-1]


