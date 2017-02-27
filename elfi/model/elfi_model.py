from functools import partial
import copy
import uuid

from elfi.utils import scipy_from_str, observed_name

from elfi.fn_wrappers import rvs_wrapper, discrepancy_wrapper
from elfi.native_client import Client
from elfi.graphical_model import GraphicalModel
from elfi.model.extensions import ScipyLikeDistribution


__all__ = ['ElfiModel', 'Constant', 'Prior', 'Simulator', 'Summary', 'Discrepancy',
           'get_current_model', 'reset_current_model']


_current_model = None


def get_current_model():
    global _current_model
    if _current_model is None:
        _current_model = ElfiModel()
    return _current_model


def reset_current_model(model=None):
    global _current_model
    if model is None:
        model = ElfiModel()
    if not isinstance(model, ElfiModel):
        raise ValueError('{} is not an instance of ElfiModel'.format(ElfiModel))
    _current_model = model


class ComputationContext():
    def __init__(self, seed=None, batch_size=None, observed=None):
        self.seed = seed
        self.batch_size = batch_size or 100
        self.observed = observed or {}


class ElfiModel(GraphicalModel):
    def __init__(self):
        self.computation_context = ComputationContext()
        self.parameter_names = []
        super(ElfiModel, self).__init__()

    def get_reference(self, name):
        cls = self.get_node(name)['class']
        return cls.reference(name, self)

    def __getitem__(self, node_name):
        return self.get_reference(node_name)

    @property
    def observed(self):
        return self.computation_context.observed

    def __copy__(self):
        model_copy = super(ElfiModel, self).__copy__()
        model_copy.computation_context = copy.copy(self.computation_context)
        model_copy.parameter_names = list(self.parameter_names)
        return model_copy


class NodeReference:
    """This is a base class for reference objects to nodes that a user of Elfi will
    typically use, e.g. `elfi.Prior` or `elfi.Simulator`. Each node has a state that
    describes how the node ultimately produces its output. The state is located in the
    ElfiModel so that serializing the model is straightforward. NodeReference and it's
    subclasses are convenience classes that makes it easy to manipulate the state and only
    contain a reference to the corresponding state in the ElfiModel.

    Currently NodeReference objects have two responsibilities:

    1. Provide convenience methods for manipulating and creating the state dictionaries of
       different types of nodes, e.g. creating a simulator node with
       `elfi.Simulator(fn, arg1, ...).
    2. Provide a compiler function that turns a state dictionary into to an Elfi
       callable output function in the computation graph. The output function will receive
       the values of its parents as arguments. The edge names correspond to argument
       names. Integers are interpreted as positional arguments. See computation graph
       for more information.

    The state of a node is a Python dictionary. It describes the type of the node and
    any other relevant state information, such as a user provided function in the case of
    elfi.Simulator.

    There are a few reserved keywords for the state dict that serve as flags for the Elfi
    compiler for specific purposes. Currently these are:

    - stochastic
        Indicates that the node is stochastic. Elfi will provide a random_state argument
        for such nodes, which contains a RandomState object for drawing random quantities.
        This node will appear in the computation graph. Using Elfi provided random states
        makes it possible to have repeatable experiments in Elfi.
    - observable
        Indicates that the true value of the node is observable. Elfi will create a copy
        of the node to the computation graph. When the user provides the observed value
        that will be added as its output. Note that if the parent observed values are
        defined, the child will be able to compute its value automatically.
    - uses_batch_size
        The node requires batch_size as input. A corresponding edge will be added to the
        computation graph.
    - uses_observed
        The node requires the observed data of its parents as input. Elfi will gather
        the observed values of its parents to a tuple and link them to the node as a named
        argument observed.
    """

    def __init__(self, name, *parents, state=None, model=None):
        """

        Parameters
        ----------
        name : string
        parents : variable
        state : dict
        model : ElfiModel
        """
        state = state or {}
        state["class"] = self.__class__
        model = model or get_current_model()

        model.add_node(name, state)

        self._init_reference(name, model)
        self._add_parents(parents)

    def _add_parents(self, parents):
        for parent in parents:
            if not isinstance(parent, NodeReference):
                parent_name = "_{}_{}".format(self.name, str(uuid.uuid4().hex[0:6]))
                parent = Constant(parent_name, parent, model=self.model)
            self.model.add_edge(parent.name, self.name)

    @classmethod
    def reference(cls, name, model):
        """Creates a reference for an existing node

        Parameters
        ----------
        name : string
            name of the node
        model : ElfiModel

        Returns
        -------
        NodePointer instance
        """
        instance = cls.__new__(cls)
        instance._init_reference(name, model)
        return instance

    def _init_reference(self, name, model):
        """Initializes all internal variables of the instance

        Parameters
        ----------
        name : name of the node in the model
        model : ElfiModel

        """
        self.name = name
        self.model = model

    def generate(self, n=1, with_values=None):
        result = Client.generate(self.model, n, self.name, with_values)
        return result[self.name]['output']

    @staticmethod
    def compile_output(state):
        return state['fn']

    def __getitem__(self, item):
        """

        Returns
        -------
        item from the state dict of the node
        """
        return self.model.get_node(self.name)[item]

    def __str__(self):
        return "{}('{}')".format(self.__class__.__name__, self.name)

    def __repr__(self):
        return self.__str__()


class Constant(NodeReference):
    def __init__(self, name, value, **kwargs):
        state = dict(value=value)
        super(Constant, self).__init__(name, state=state, **kwargs)

    @staticmethod
    def compile_output(state):
        return state['value']


class StochasticMixin(NodeReference):
    def __init__(self, *args, state, **kwargs):
        # Flag that this node is stochastic
        state['stochastic'] = True
        super(StochasticMixin, self).__init__(*args, state=state, **kwargs)


class ObservableMixin(NodeReference):
    """
    """

    def __init__(self, *args, state, observed=None, **kwargs):
        # Flag that this node can be observed
        state['observable'] = True
        super(ObservableMixin, self).__init__(*args, state=state, **kwargs)

        if observed is not None:
            self.model.computation_context.observed[self.name] = observed

    @property
    def observed(self):
        obs_name = observed_name(self.name)
        result = Client.generate(self.model, 0, obs_name)
        return result[obs_name]['output']


class ScipyLikeRV(StochasticMixin, NodeReference):
    def __init__(self, name, distribution="uniform", *params, size=1, **kwargs):
        """

        Parameters
        ----------
        name : str
        distribution : str or scipy-like distribution object
        params : params of the distribution
        size : int, tuple or None
            size of a single random draw. None means a scalar.

        """

        state = dict(distribution=distribution, size=size, require=('batch_size',))
        super(ScipyLikeRV, self).__init__(name, *params, state=state, **kwargs)

    @staticmethod
    def compile_output(state):
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
        return output

    @property
    def size(self):
        return self['size']

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

        return super(ScipyLikeRV, self).__str__()[0:-1] + ", {})".format(name)


class Prior(ScipyLikeRV):
    pass


class Simulator(StochasticMixin, ObservableMixin, NodeReference):
    def __init__(self, name, fn, *dependencies, **kwargs):
        state = dict(fn=fn, require=('batch_size',))
        super(Simulator, self).__init__(name, *dependencies, state=state, **kwargs)


class Summary(ObservableMixin, NodeReference):
    def __init__(self, name, fn, *dependencies, **kwargs):
        state = dict(fn=fn)
        super(Summary, self).__init__(name, *dependencies, state=state, **kwargs)


class Discrepancy(NodeReference):
    def __init__(self, name, fn, *dependencies, **kwargs):
        state = dict(fn=fn, require=('observed',))
        super(Discrepancy, self).__init__(name, *dependencies, state=state, **kwargs)

    @staticmethod
    def compile_output(state):
        fn = state['fn']
        output_fn = partial(discrepancy_wrapper, fn=fn)
        return output_fn












