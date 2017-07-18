import logging
import copy
import inspect
import re
import uuid
from functools import partial
import logging

import scipy.spatial

import elfi.client
from elfi.model.graphical_model import GraphicalModel
from elfi.model.utils import rvs_from_distribution, distance_as_discrepancy
from elfi.store import OutputPool
from elfi.utils import scipy_from_str, observed_name, random_seed

logger = logging.getLogger(__name__)

__all__ = ['ElfiModel', 'ComputationContext', 'NodeReference',
           'Constant', 'Operation', 'RandomVariable',
           'Prior', 'Simulator', 'Summary', 'Discrepancy', 'Distance',
           'get_current_model', 'set_current_model', 'new_model']


logger = logging.getLogger(__name__)


"""This module contains the classes for creating generative models in ELFI. The class that
describes the generative model is named `ElfiModel`."""


_current_model = None


def get_current_model():
    """Return the current default `elfi.ElfiModel` instance.

    New nodes will be added to this model by default.
    """
    global _current_model
    if _current_model is None:
        _current_model = ElfiModel()
    return _current_model


def set_current_model(model=None):
    """Set the current default `elfi.ElfiModel` instance."""
    global _current_model
    if model is None:
        model = ElfiModel()
    if not isinstance(model, ElfiModel):
        raise ValueError('{} is not an instance of ElfiModel'.format(ElfiModel))
    _current_model = model


def new_model(name=None, set_current=True):
    model = ElfiModel(name=name)
    if set_current:
        set_current_model(model)
    return model


def random_name(length=4, prefix=''):
    return prefix + str(uuid.uuid4().hex[0:length])


# TODO: move to another file
class ComputationContext:
    """Container object for key components for consistent computation results.

    Attributes
    ----------
    seed : int
    batch_size : int
    pool : elfi.OutputPool
    num_submissions : int
        Number of submissions using this context.

    Notes
    -----
    The attributes are immutable.

    """
    def __init__(self, batch_size=None, seed=None, pool=None):
        """

        Parameters
        ----------
        batch_size : int
        seed : int, None, 'global'
            When None generates a random integer seed. When `'global'` uses the global
            numpy random state. Only recommended for debugging
        pool : elfi.OutputPool

        """

        # Check pool context
        if pool is not None and pool.has_context:
            if batch_size is None:
                batch_size = pool.batch_size
            elif batch_size != pool.batch_size:
                raise ValueError('Pool batch_size differs from the given batch_size!')

            if seed is None:
                seed = pool.seed
            elif seed != pool.seed:
                raise ValueError('Pool seed differs from the given seed!')

        self._batch_size = batch_size or 1
        self._seed = random_seed() if seed is None else seed
        self._pool = pool

        # Count the number of submissions from this context
        self.num_submissions = 0

        if pool is not None and not pool.has_context:
            self._pool.set_context(self)

    @property
    def pool(self):
        return self._pool

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def seed(self):
        return self._seed

    def callback(self, batch, batch_index):
        if self._pool is not None:
            self._pool.add_batch(batch, batch_index)


class ElfiModel(GraphicalModel):
    """A generative model for LFI
    """
    def __init__(self, name=None, observed=None, source_net=None):
        """

        Parameters
        ----------
        name : str, optional
        observed : dict, optional
            Observed data with node names as keys.
        source_net : nx.DiGraph, optional
        set_current : bool, optional
            Sets this model as the current ELFI model
        """

        super(ElfiModel, self).__init__(source_net)
        self.name = name or "model_{}".format(random_name())
        self.observed = observed or {}

    @property
    def name(self):
        """Name of the model"""
        return self.source_net.graph['name']

    @name.setter
    def name(self, name):
        """Sets the name of the model"""
        self.source_net.graph['name'] = name

    @property
    def observed(self):
        """The observed data for the nodes in a dictionary."""
        return self.source_net.graph['observed']

    @observed.setter
    def observed(self, observed):
        """Set the observed data of the model

        Parameters
        ----------
        observed : dict

        """
        if not isinstance(observed, dict):
            raise ValueError("Observed data must be given in a dictionary with the node"
                             "name as the key")
        self.source_net.graph['observed'] = observed

    def generate(self, batch_size=1, outputs=None, with_values=None):
        """Generates a batch of outputs using the global seed.

        This method is useful for testing that the generative model works.

        Parameters
        ----------
        batch_size : int
        outputs : list
        with_values : dict
            You can specify values for nodes to use when generating data

        """

        if outputs is None:
            outputs = self.source_net.nodes()
        elif isinstance(outputs, str):
            outputs = [outputs]
        if not isinstance(outputs, list):
            raise ValueError('Outputs must be a list of node names')

        pool = None
        if with_values is not None:
            pool = OutputPool(with_values.keys())
            pool.add_batch(with_values, 0)
        context = ComputationContext(batch_size, seed='global', pool=pool)

        client = elfi.client.get_client()
        compiled_net = client.compile(self.source_net, outputs)
        loaded_net = client.load_data(compiled_net, context, batch_index=0)
        return client.compute(loaded_net)

    def get_reference(self, name):
        """Returns a new reference object for a node in the model."""
        cls = self.get_node(name)['_class']
        return cls.reference(name, self)

    def get_state(self, name):
        """Return the state of the node."""
        return self.source_net.node[name]

    def update_node(self, name, updating_name):
        """Updates `node` with `updating_node` in the model.

        The node with name `name` gets the state (operation), parents and observed
        data (if applicable) of the updating_node. The updating node is then removed
        from the graph.

        Parameters
        ----------
        name : str
        updating_name : str
        """

        update_observed = False
        obs = None
        if updating_name in self.observed:
            update_observed = True
            obs = self.observed.pop(updating_name)

        super(ElfiModel, self).update_node(name, updating_name)

        # Move data to the updated node
        if update_observed:
            self.observed[name] = obs

    def remove_node(self, name):
        """Remove a node from the graph

        Parameters
        ----------
        name : str

        """
        if name in self.observed:
            self.observed.pop(name)
        super(ElfiModel, self).remove_node(name)

    @property
    def parameter_names(self):
        """A list of model parameter names in an alphabetical order."""
        return sorted([n for n in self.nodes if '_parameter' in self.get_state(n)])

    @parameter_names.setter
    def parameter_names(self, parameter_names):
        """Set the model parameter nodes.
        
        For each node name in parameters, the corresponding node will be marked as being a
        parameter node. Other nodes will be marked as not being parameter nodes.
        
        Parameters
        ----------
        parameter_names : list
            A list of parameter names
        
        Returns
        -------
        None
        """
        parameter_names = set(parameter_names)
        for n in self.nodes:
            state = self.get_state(n)
            if n in parameter_names:
                parameter_names.remove(n)
                state['_parameter'] = True
            else:
                if '_parameter' in state: state.pop('_parameter')
        if len(parameter_names) > 0:
            raise ValueError('Parameters {} not found from the model'.format(parameter_names))

    def copy(self):
        """Return a copy of the ElfiModel instance

        Returns
        -------
        ElfiModel

        """
        kopy = super(ElfiModel, self).copy()
        kopy.name = "{}_copy_{}".format(self.name, random_name())
        return kopy

    def __getitem__(self, node_name):
        return self.get_reference(node_name)


class InstructionsMapper:
    @property
    def state(self):
        raise NotImplementedError()

    @property
    def uses_meta(self):
        return self.state.get('_uses_meta', False)

    @uses_meta.setter
    def uses_meta(self, val):
        self.state['_uses_meta'] = True


class NodeReference(InstructionsMapper):
    """A base class for node objects in the model.

    A user of ELFI will typically use, e.g. `elfi.Prior` or `elfi.Simulator` to create
    state dictionaries for nodes.

    Each node has a state dictionary that describes how the node ultimately produces its
    output (see module documentation for more details). The state is stored in the
    `ElfiModel` so that serializing the model is straightforward. `NodeReference` and it's
    subclasses are convenience classes that make it easy to manipulate the state. They
    only contain a reference to the corresponding state in the `ElfiModel`.

    Examples
    --------
    ::

        elfi.Simulator(fn, arg1, ...)

    creates a node to `self.model.source_net` with the following state dictionary::

        dict(_operation=fn, _class=elfi.Simulator, ...)

    and adds and edge from arg1 to to the new simulator node in the
    `self.model.source_net`.

    """
    def __init__(self, *parents, state=None, model=None, name=None):
        """

        Parameters
        ----------
        parents : variable
        name : string
            If name ends in an asterisk '*' character, the asterisk will be replaced with
            a random string and the name is ensured to be unique within the model.
        state : dict
        model : elfi.ElfiModel

        Examples
        --------
        >>> node = NodeReference(name='name*') # doctest: +SKIP
        >>> node.name # doctest: +SKIP
        name_1f4rgh

        """
        state = state or {}
        state['_class'] = self.__class__
        model = self._determine_model(model, parents)

        name = self._give_name(name, model)
        model.add_node(name, state)

        self._init_reference(name, model)
        self._add_parents(parents)

    def _add_parents(self, parents):
        for parent in parents:
            if not isinstance(parent, NodeReference):
                parent_name = self._new_name('_' + self.name)
                parent = Constant(parent, name=parent_name, model=self.model)
            self.model.add_edge(parent.name, self.name)

    def _determine_model(self, model, parents):
        if not isinstance(model, ElfiModel) and model is not None:
            return ValueError('Invalid model passed {}'.format(model))

        # Check that parents belong to the same model and inherit the model if needed
        for p in parents:
            if isinstance(p, NodeReference):
                if model is None:
                    model = p.model
                elif model != p.model:
                    raise ValueError('Parents are from different models!')

        if model is None:
            model = get_current_model()

        return model

    @property
    def parents(self):
        """Get all the positional parent nodes (inputs) of this node

        Returns
        -------
        parents : list
            List of positional parents
        """
        return [self.model[p] for p in self.model.get_parents(self.name)]

    @classmethod
    def reference(cls, name, model):
        """Constructor for creating a reference for an existing node in the model

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

    def become(self, other_node):
        """Make this node become the `other_node`.

        The children of this node will be preserved.

        Parameters
        ----------
        other_node : NodeReference

        """
        if other_node.model is not self.model:
            raise ValueError('The other node belongs to a different model')

        self.model.update_node(self.name, other_node.name)

        # Update the reference class
        _class = self.state.get('_class', NodeReference)
        if not isinstance(self, _class):
            self.__class__ = _class

        # Update also the other node reference
        other_node.name = self.name
        other_node.model = self.model

    def _init_reference(self, name, model):
        """Initializes all internal variables of the instance

        Parameters
        ----------
        name : name of the node in the model
        model : ElfiModel

        """
        self.name = name
        self.model = model

    def generate(self, batch_size=1, with_values=None):
        """Generates output from this node.

        Useful for testing.

        Parameters
        ----------
        batch_size : int
        with_values : dict

        """
        result = self.model.generate(batch_size, self.name, with_values=with_values)
        return result[self.name]

    def _give_name(self, name, model):
        if name is not None:
            if name[-1] == '*':
                # Generate unique name
                name = self._new_name(name[:-1], model)
            return name

        try:
            name = self._inspect_name()
        except:
            logger.warning("Automatic name inspection failed, using a random name "
                           "instead. This may be caused by using an interactive Python "
                           "shell. You can provide a name parameter e.g. "
                           "elfi.Prior('uniform', name='nodename') to suppress this "
                           "warning.")
            name = None

        if name is None or model.has_node(name):
            name = self._new_name(model=model)

        return name

    def _inspect_name(self):
        """Magic method in trying to infer the name from the code.

        Does not work in interactive python shell."""

        # Test if context info is available and try to give the same name as the variable
        # Please note that this is only a convenience method which is not guaranteed to
        # work in all cases. If you require a specific name, pass the name argument.
        frame = inspect.currentframe()
        if frame is None:
            return None

        # Frames are available
        # Take the callers frame
        frame = frame.f_back.f_back.f_back
        info = inspect.getframeinfo(frame, 1)

        # Skip super calls to find the assignment frame
        while re.match('\s*super\(', info.code_context[0]):
            frame = frame.f_back
            info = inspect.getframeinfo(frame, 1)

        # Match simple direct assignment with the class name, no commas or semicolons
        # Also do not accept a name starting with an underscore
        rex = '\s*([^\W_][\w]*)\s*=\s*\w?[\w\.]*{}\('.format(self.__class__.__name__)
        match = re.match(rex, info.code_context[0])
        if match:
            name = match.groups()[0]
            return name
        else:
            return None

    def _new_name(self, basename='', model=None):
        model = model or self.model
        if not basename:
            basename = '_{}'.format(self.__class__.__name__.lower())
        while True:
            name = "{}_{}".format(basename, random_name())
            if not model.has_node(name): break
        return name

    @property
    def state(self):
        """State dictionary of the node"""
        if self.model is None:
            raise ValueError('{} {} is not initialized'.format(self.__class__.__name__,
                                                               self.name))
        return self.model.get_node(self.name)

    def __getitem__(self, item):
        """Get item from the state dict of the node
        """
        return self.state[item]

    def __setitem__(self, item, value):
        """Set item into the state dict of the node
        """
        self.state[item] = value

    def __repr__(self):
        return "{}(name='{}')".format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.name


class StochasticMixin(NodeReference):
    """Makes a node stochastic

    Operations of stochastic nodes will receive a `random_state` keyword argument.
    """
    def __init__(self, *parents, state, **kwargs):
        # Flag that this node is stochastic
        state['_stochastic'] = True
        super(StochasticMixin, self).__init__(*parents, state=state, **kwargs)


class ObservableMixin(NodeReference):
    """Makes a node observable

    Observable nodes accept observed keyword argument. In addition the compiled
    model will contain a sister node that contains the observed value or will compute the
    observed value from the observed values of it's parents.
    """

    def __init__(self, *parents, state, observed=None, **kwargs):
        # Flag that this node can be observed
        state['_observable'] = True
        super(ObservableMixin, self).__init__(*parents, state=state, **kwargs)

        # Set the observed value
        if observed is not None:
            self.model.observed[self.name] = observed

    @property
    def observed(self):
        obs_name = observed_name(self.name)
        result = self.model.generate(0, obs_name)
        return result[obs_name]


# User interface nodes


class Constant(NodeReference):
    """A node holding a constant value."""
    def __init__(self, value, **kwargs):
        state = dict(_output=value)
        super(Constant, self).__init__(state=state, **kwargs)


class Operation(NodeReference):
    """A generic deterministic operation node.
    """
    def __init__(self, fn, *parents, **kwargs):
        state = dict(_operation=fn)
        super(Operation, self).__init__(*parents, state=state, **kwargs)


class RandomVariable(StochasticMixin, NodeReference):
    """A node that draws values from a random distribution."""

    def __init__(self, distribution, *params, size=None, **kwargs):
        """

        Parameters
        ----------
        distribution : str or scipy-like distribution object
        params : params of the distribution
        size : int, tuple or None, optional
            Output size of a single random draw.

        """

        state = dict(distribution=distribution,
                     size=size,
                     _uses_batch_size=True)
        state['_operation'] = self.compile_operation(state)
        super(RandomVariable, self).__init__(*params, state=state, **kwargs)

    @staticmethod
    def compile_operation(state):
        size = state['size']
        distribution = state['distribution']
        if not (size is None or isinstance(size, tuple)):
            size = (size, )

        # Note: sending the scipy distribution object also pickles the global numpy random
        # state with it. If this needs to be avoided, the object needs to be constructed
        # on the worker.
        if isinstance(distribution, str):
            distribution = scipy_from_str(distribution)

        if not hasattr(distribution, 'rvs'):
            raise ValueError("Distribution {} "
                             "must implement a rvs method".format(distribution))

        op = partial(rvs_from_distribution, distribution=distribution, size=size)
        return op

    @property
    def distribution(self):
        """Returns the distribution object."""
        distribution = self['distribution']
        if isinstance(distribution, str):
            distribution = scipy_from_str(distribution)
        return distribution

    @property
    def size(self):
        """Returns the size of the output from the distribution."""
        return self['size']

    def __repr__(self):
        d = self.distribution

        if isinstance(d, str):
            name = "'{}'".format(d)
        elif hasattr(d, 'name'):
            name = "'{}'".format(d.name)
        elif isinstance(d, type):
            name = d.__name__
        else:
            name = d.__class__.__name__

        return super(RandomVariable, self).__repr__()[0:-1] + ", {})".format(name)


class Prior(RandomVariable):
    """A parameter node of a generative model."""
    def __init__(self, distribution, *params, size=None, **kwargs):
        """

        Parameters
        ----------
        distribution : str, object
            Any distribution from `scipy.stats`, either as a string or an object. Objects
            must implement at least an `rvs` method with signature
            `rvs(*parameters, size, random_state)`. Can also be a custom distribution
            object that implements at least an `rvs` method. Many of the algorithms also
            require the `pdf` and `logpdf` methods to be available.
        size : int, tuple or None, optional
            Output size of a single random draw.
        params
            Parameters of the prior distribution
        kwargs

        Notes
        -----
        The parameters of the `scipy` distributions (typically `loc` and `scale`) must be
        given as positional arguments.

        Many algorithms (e.g. SMC) also require a `pdf` method for the distribution. In
        general the definition of the distribution is a subset of
        `scipy.stats.rv_continuous`.

        Scipy distributions: https://docs.scipy.org/doc/scipy-0.19.0/reference/stats.html

        """
        super(Prior, self).__init__(distribution, *params, size=size, **kwargs)
        self['_parameter'] = True


class Simulator(StochasticMixin, ObservableMixin, NodeReference):
    """A simulator node of a generative model.

    Simulator nodes are stochastic and may have observed data in the model.
    """
    def __init__(self, fn, *params, **kwargs):
        """

        Parameters
        ----------
        fn : callable
            Simulator function with a signature `sim(*params, batch_size, random_state)`
        params
            Input parameters for the simulator.
        kwargs
        """
        state = dict(_operation=fn, _uses_batch_size=True)
        super(Simulator, self).__init__(*params, state=state, **kwargs)


class Summary(ObservableMixin, NodeReference):
    """A summary node of a generative model.

    Summary nodes are deterministic operations associated with the observed data. if their
    parents hold observed data it will be automatically transformed.
    """
    def __init__(self, fn, *parents, **kwargs):
        """

        Parameters
        ----------
        fn : callable
            Summary function with a signature `summary(*parents)`
        parents
            Input data for the summary function.
        kwargs
        """
        if not parents:
            raise ValueError('This node requires that at least one parent is specified.')
        state = dict(_operation=fn)
        super(Summary, self).__init__(*parents, state=state, **kwargs)


class Discrepancy(NodeReference):
    """A discrepancy node of a generative model.

    This class provides a convenience node for custom distance operations.
    """
    def __init__(self, discrepancy, *parents, **kwargs):
        """

        Parameters
        ----------
        discrepancy : callable
            Signature of the discrepancy function is of the form:
            `discrepancy(summary_1, summary_2, ..., observed)`, where summaries are
            arrays containing `batch_size` simulated values and observed is a tuple
            (observed_summary_1, observed_summary_2, ...). The callable object should
            return a vector of discrepancies between the simulated summaries and the
            observed summaries.
        *parents
            Typically the summaries for the discrepancy function.
        **kwargs

        See Also
        --------
        elfi.Distance : creating common distance discrepancies.

        """
        if not parents:
            raise ValueError('This node requires that at least one parent is specified.')
        state = dict(_operation=discrepancy, _uses_observed=True)
        super(Discrepancy, self).__init__(*parents, state=state, **kwargs)


# TODO: add weights
class Distance(Discrepancy):
    def __init__(self, distance, *summaries, p=None, w=None, V=None, VI=None, **kwargs):
        """A distance node of a generative model.

        This class contains many common distance implementations through scipy.

        Parameters
        ----------
        distance : str, callable
            If string it must be a valid metric from `scipy.spatial.distance.cdist`.

            Is a callable, the signature must be `distance(X, Y)`, where X is a n x m
            array containing n simulated values (summaries) in rows and Y is a 1 x m array
            that contains the observed values (summaries). The callable should return
            a vector of distances between the simulated summaries and the observed
            summaries.
        summaries
            summary nodes of the model
        p : double, optional
            The p-norm to apply Only for distance Minkowski (`'minkowski'`), weighted
            and unweighted. Default: 2.
        w : ndarray, optional
            The weight vector. Only for weighted Minkowski (`'wminkowski'`). Mandatory.
        V : ndarray, optional
            The variance vector. Only for standardized Euclidean (`'seuclidean'`).
            Mandatory.
        VI : ndarray, optional
            The inverse of the covariance matrix. Only for Mahalanobis. Mandatory.

        Examples
        --------
        >>> d = elfi.Distance('euclidean', summary1, summary2...) # doctest: +SKIP

        >>> d = elfi.Distance('minkowski', summary, p=1) # doctest: +SKIP

        Notes
        -----
        Your summaries need to be scalars or vectors for this method to work. The
        summaries will be first stacked to a single 2D array with the simulated
        summaries in the rows for every simulation and the distance is taken row
        wise against the corresponding observed summary vector.

        Scipy distances: https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.spatial.distance.cdist.html

        See Also
        --------
        elfi.Discrepancy : A general discrepancy node

        """
        if not summaries:
            raise ValueError("This node requires that at least one parent is specified.")

        if isinstance(distance, str):
            if distance == 'wminkowski' and w is None:
                raise ValueError('Parameter w must be specified for distance=wminkowski.')
            if distance == 'seuclidean' and V is None:
                raise ValueError('Parameter V must be specified for distance=seuclidean.')
            if distance == 'mahalanobis' and VI is None:
                raise ValueError('Parameter VI must be specified for distance=mahalanobis.')
            cdist_kwargs = dict(metric=distance, p=p, w=w, V=V, VI=VI)
            dist_fn = partial(scipy.spatial.distance.cdist, **cdist_kwargs)
        else:
            dist_fn = distance
        discrepancy = partial(distance_as_discrepancy, dist_fn)
        super(Distance, self).__init__(discrepancy, *summaries, **kwargs)
        # Store the original passed distance
        self.state['distance'] = distance
