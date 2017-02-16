from functools import partial
import uuid

from elfi.utils import scipy_from_str

from elfi.fn_wrappers import rvs_wrapper, discrepancy_wrapper
from elfi.native_client import Client
from elfi.network import Network
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


class ElfiModel(Network):
    def __init__(self):
        self.observed = {}
        self.parameter_names = []
        super(ElfiModel, self).__init__()

    def get_reference(self, name):
        cls = self.get_node(name)['class']
        return cls.reference(name, self)


class NodeReference:

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
            self.model.observed[self.name] = observed

    @property
    def observed(self):
        # TODO: check that no stochastic nodes are executed?
        obs_name = "_{}_observed".format(self.name)
        result = Client.generate(self.model, 0, obs_name)
        return result[obs_name]['output']


class ScipyLikeRV(NodeReference):
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


class Simulator(ObservableMixin, NodeReference):
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












