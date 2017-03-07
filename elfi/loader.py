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


class OutputSupplyLoader(Loader):

    @classmethod
    def load(cls, context, output_net, batch_index):
        for node, supply in context.output_supply.items():
            output_net.node[node]['output'] = supply[batch_index][node]
        return output_net

class RandomStateLoader(Loader):
    """
    Add random state instance for the node
    """

    @classmethod
    def load(cls, context, output_net, batch_index):
        seed = context.seed
        if seed is False:
            # Use the global numpy random_state
            random_state = None
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


