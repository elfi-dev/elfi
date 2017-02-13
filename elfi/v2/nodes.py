from functools import partial
import uuid

from elfi.v2.utils import scipy_from_str

from elfi.v2.fn_wrappers import rvs_wrapper, discrepancy_wrapper
from elfi.v2.native_client import Client
from elfi.v2.elfi_model import ElfiModel, get_current_model


class NodeReference:

    def __init__(self, name, *parents, state=None, network=None):
        """

        Parameters
        ----------
        name : string
        parents : variable
        state : dict
        network : ElfiModel
        """
        state = state or {}
        state["class"] = self.__class__
        network = network or get_current_model()

        network.add_node(name, state)

        self._init_reference(name, network)
        self._add_parents(parents)

    def _add_parents(self, parents):
        for parent in parents:
            if not isinstance(parent, NodeReference):
                parent_name = "_{}_{}".format(self.name, str(uuid.uuid4().hex[0:6]))
                parent = Constant(parent_name, parent, network=self.network)
            self.network.add_edge(parent.name, self.name)

    @classmethod
    def reference(cls, name, network):
        """Creates a reference for an existing node

        Returns
        -------
        NodePointer instance
        """
        instance = cls.__new__(cls)
        instance._init_reference(name, network)
        return instance

    def _init_reference(self, name, network):
        """Initializes all internal variables of the instance

        Parameters
        ----------
        name : name of the node in the network
        network : ElfiModel

        """
        self.name = name
        self.network = network

    def generate(self, n=1, with_values=None):
        result = Client.generate(self.network, n, self.name, with_values)
        return result[self.name]['output']

    def __getitem__(self, item):
        """

        Returns
        -------
        item from the state dict of the node
        """
        return self.network.get_node(self.name)[item]

    def __str__(self):
        return "{}('{}')".format(self.__class__.__name__, self.name)

    def __repr__(self):
        return self.__str__()


class Constant(NodeReference):
    def __init__(self, name, value, **kwargs):
        state = {
            "value": value,
        }
        super(Constant, self).__init__(name, state=state, **kwargs)

    @staticmethod
    def compile(state):
        return dict(output=state['value'])


class ObservedMixin(NodeReference):
    """
    """

    def __init__(self, *args, state, observed=None, **kwargs):
        # Flag that this node can be observed
        state['observable'] = True
        super(ObservedMixin, self).__init__(*args, state=state, **kwargs)

        if observed is not None:
            self.network.observed[self.name] = observed

    @property
    def observed(self):
        # TODO: check that no stochastic nodes are executed
        # Generate this way to avoid cost of simulation
        obs_name = "_{}_observed".format(self.name)
        result = Client.generate(self.network, 0, obs_name)
        return result[obs_name]['output']


class RandomVariable(NodeReference):
    def __init__(self, name, distribution="uniform", *params, size=None, **kwargs):
        state = dict(distribution=distribution, size=size)
        super(RandomVariable, self).__init__(name, *params, state=state, **kwargs)

    @staticmethod
    def compile(state):
        size = state['size']
        distribution = state['distribution']
        if not (size is None or isinstance(size, tuple)):
            size = (size, )

        if isinstance(distribution, str):
            distribution = scipy_from_str(distribution)

        if not hasattr(distribution, 'rvs'):
            raise ValueError("Distribution {} "
                             "must implement a rvs method".format(distribution))

        output = partial(rvs_wrapper, distribution=distribution, size=size)
        return dict(output=output, runtime=('batch_size',))

    def __str__(self):
        d = self['distribution']

        if isinstance(d, str):
            name = "'{}'".format(d)
        elif hasattr(d, 'name'):
            name = "'{}'".format(d.name)
        elif isinstance(d, type):
            name = d.__name__
        else:
            name = d.__class__.__name__

        return super(RandomVariable, self).__str__()[0:-1] + ", {})".format(name)


class Prior(RandomVariable):
    pass


class Simulator(ObservedMixin, NodeReference):
    def __init__(self, name, fn, *dependencies, **kwargs):
        state = dict(fn=fn)
        super(Simulator, self).__init__(name, *dependencies, state=state, **kwargs)

    @staticmethod
    def compile(state):
        fn = state['fn']
        return dict(output=fn, runtime=('batch_size',))


class Summary(ObservedMixin, NodeReference):
    def __init__(self, name, fn, *dependencies, **kwargs):
        state = dict(fn=fn)
        super(Summary, self).__init__(name, *dependencies, state=state, **kwargs)

    @staticmethod
    def compile(state):
        fn = state['fn']
        return dict(output=fn)


class Discrepancy(NodeReference):
    def __init__(self, name, fn, *dependencies, **kwargs):
        state = dict(fn=fn)
        super(Discrepancy, self).__init__(name, *dependencies, state=state, **kwargs)

    @staticmethod
    def compile(state):
        fn = state['fn']
        output_fn = partial(discrepancy_wrapper, fn=fn)
        return dict(output=output_fn, require=('observed',))

















