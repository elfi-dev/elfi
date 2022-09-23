"""This module contains classes for creating ELFI graphs (`ElfiModel`).

The ElfiModel is a directed acyclic graph (DAG), whose nodes represent
parts of the inference task, for example the parameters to be inferred,
the simulator or a summary statistic.

https://en.wikipedia.org/wiki/Directed_acyclic_graph
"""

import inspect
import logging
import os
import pickle
import re
import uuid
from functools import partial

import numpy as np
import scipy.spatial

import elfi.client
from elfi.model.graphical_model import GraphicalModel
from elfi.model.utils import distance_as_discrepancy, rvs_from_distribution
from elfi.store import OutputPool
from elfi.utils import observed_name, random_seed, scipy_from_str

__all__ = [
    'ElfiModel', 'ComputationContext', 'NodeReference', 'Constant',
    'Operation', 'RandomVariable', 'Prior', 'Simulator', 'Summary',
    'Discrepancy', 'Distance', 'AdaptiveDistance', 'get_default_model',
    'set_default_model', 'new_model', 'load_model'
]

logger = logging.getLogger(__name__)
_default_model = None


def get_default_model():
    """Return the current default ``ElfiModel`` instance.

    New nodes will be added to this model by default.
    """
    global _default_model
    if _default_model is None:
        _default_model = ElfiModel()
    return _default_model


def set_default_model(model=None):
    """Set the current default ``ElfiModel`` instance.

    New nodes will be placed the given model by default.

    Parameters
    ----------
    model : ElfiModel, optional
        If None, creates a new ``ElfiModel``.

    """
    global _default_model
    if model is None:
        model = ElfiModel()
    if not isinstance(model, ElfiModel):
        raise ValueError('{} is not an instance of ElfiModel'.format(ElfiModel))
    _default_model = model


def new_model(name=None, set_default=True):
    """Create a new ``ElfiModel`` instance.

    In addition to making a new ElfiModel instance, this method sets the new instance as
    the default for new nodes.

    Parameters
    ----------
    name : str, optional
    set_default : bool, optional
        Whether to set the newly created model as the current model.

    """
    model = ElfiModel(name=name)
    if set_default:
        set_default_model(model)
    return model


def load_model(name, prefix=None, set_default=True):
    """Load the pickled ElfiModel.

    Assumes there exists a file "name.pkl" in the current directory. Also sets the loaded
    model as the default model for new nodes.

    Parameters
    ----------
    name : str
        Name of the model file to load (without the .pkl extension).
    prefix : str
        Path to directory where the model file is located, optional.
    set_default : bool, optional
        Set the loaded model as the default model. Default is True.

    Returns
    -------
    ElfiModel

    """
    model = ElfiModel.load(name, prefix=prefix)
    if set_default:
        set_default_model(model)
    return model


def random_name(length=4, prefix=''):
    """Generate a random string.

    Parameters
    ----------
    length : int, optional
    prefix : str, optional

    """
    return prefix + str(uuid.uuid4().hex[0:length])


# TODO: move to another file?
class ComputationContext:
    """Container object for key components for consistent computation results.

    Attributes
    ----------
    seed : int
    batch_size : int
    pool : OutputPool
    num_submissions : int
        Number of submissions using this context.
    sub_seed_cache : dict
        Caches the sub seed generation state variables. This is

    Notes
    -----
    The attributes are immutable.

    """

    def __init__(self, batch_size=None, seed=None, pool=None):
        """Set up a ComputationContext.

        Parameters
        ----------
        batch_size : int, optional
        seed : int, None, 'global', optional
            When None generates a random integer seed. When `'global'` uses the global
            numpy random state. Only recommended for debugging.
        pool : elfi.OutputPool, optional
            Used for storing output.

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

        # Caches will not be used if they are not found from the caches dict
        self.caches = {'executor': {}, 'sub_seed': {}}

        # Count the number of submissions from this context
        self.num_submissions = 0

        if pool is not None and not pool.has_context:
            self._pool.set_context(self)

    @property
    def pool(self):
        """Return the output pool."""
        return self._pool

    @property
    def batch_size(self):
        """Return the batch size."""
        return self._batch_size

    @property
    def seed(self):
        """Return the random seed."""
        return self._seed

    def callback(self, batch, batch_index):
        """Add the batch to pool.

        Parameters
        ----------
        batch : dict
        batch_index : int

        """
        if self._pool is not None:
            self._pool.add_batch(batch, batch_index)


class ElfiModel(GraphicalModel):
    """A container for the inference model.

    The ElfiModel is a directed acyclic graph (DAG), whose nodes represent
    parts of the inference task, for example the parameters to be inferred,
    the simulator or a summary statistic.
    """

    def __init__(self, name=None, observed=None, source_net=None):
        """Initialize the inference model.

        Parameters
        ----------
        name : str, optional
        observed : dict, optional
            Observed data with node names as keys.
        source_net : nx.DiGraph, optional
        set_current : bool, optional
            Sets this model as the current (default) ELFI model

        """
        super(ElfiModel, self).__init__(source_net)
        self.name = name or "model_{}".format(random_name())
        self.observed = observed or {}

    @property
    def name(self):
        """Return name of the model."""
        return self.source_net.graph['name']

    @name.setter
    def name(self, name):
        """Set the name of the model."""
        self.source_net.graph['name'] = name

    @property
    def observed(self):
        """Return the observed data for the nodes in a dictionary."""
        return self.source_net.graph['observed']

    @observed.setter
    def observed(self, observed):
        """Set the observed data of the model.

        Parameters
        ----------
        observed : dict

        """
        if not isinstance(observed, dict):
            raise ValueError("Observed data must be given in a dictionary with the node"
                             "name as the key")
        self.source_net.graph['observed'] = observed

    def generate(self, batch_size=1, outputs=None, with_values=None, seed=None):
        """Generate a batch of outputs.

        This method is useful for testing that the ELFI graph works.

        Parameters
        ----------
        batch_size : int, optional
        outputs : list, optional
        with_values : dict, optional
            You can specify values for nodes to use when generating data
        seed : int, optional
            Defaults to global numpy seed.

        """
        if outputs is None:
            outputs = list(self.source_net.nodes())
        elif isinstance(outputs, str):
            outputs = [outputs]
        if not isinstance(outputs, list):
            raise ValueError('Outputs must be a list of node names')

        if seed is None:
            seed = 'global'

        pool = None
        if with_values is not None:
            pool = OutputPool(with_values.keys())
            pool.add_batch(with_values, 0)
        context = ComputationContext(batch_size, seed=seed, pool=pool)

        client = elfi.client.get_client()
        compiled_net = client.compile(self.source_net, outputs)
        loaded_net = client.load_data(compiled_net, context, batch_index=0)
        return client.compute(loaded_net)

    def get_reference(self, name):
        """Return a new reference object for a node in the model.

        Parameters
        ----------
        name : str

        """
        cls = self.get_node(name)['attr_dict']['_class']
        return cls.reference(name, self)

    def get_state(self, name):
        """Return the state of the node.

        Parameters
        ----------
        name : str

        """
        return self.source_net.nodes[name]

    def update_node(self, name, updating_name):
        """Update `node` with `updating_node` in the model.

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
        """Remove a node from the graph.

        Parameters
        ----------
        name : str

        """
        if name in self.observed:
            self.observed.pop(name)
        super(ElfiModel, self).remove_node(name)

    @property
    def parameter_names(self):
        """Return a list of model parameter names in an alphabetical order."""
        return sorted([n for n in self.nodes if '_parameter' in self.get_state(n)['attr_dict']])

    @parameter_names.setter
    def parameter_names(self, parameter_names):
        """Set the model parameter nodes.

        For each node name in parameters, the corresponding node will be marked as being a
        parameter node. Other nodes will be marked as not being parameter nodes.

        Parameters
        ----------
        parameter_names : list
            A list of parameter names

        """
        parameter_names = set(parameter_names)
        for n in self.nodes:
            state = self.get_state(n)['attr_dict']
            if n in parameter_names:
                parameter_names.remove(n)
                state['_parameter'] = True
            else:
                if '_parameter' in state:
                    state.pop('_parameter')
        if len(parameter_names) > 0:
            raise ValueError('Parameters {} not found from the model'.format(parameter_names))

    def copy(self):
        """Return a copy of the ElfiModel instance.

        Returns
        -------
        ElfiModel

        """
        kopy = super(ElfiModel, self).copy()
        kopy.name = "{}_copy_{}".format(self.name, random_name())
        return kopy

    def save(self, prefix=None):
        """Save the current model to pickled file.

        Parameters
        ----------
        prefix : str, optional
            Path to the directory under which to save the model. Default is the current working
            directory.

        """
        path = self.name + '.pkl'
        if prefix is not None:
            os.makedirs(prefix, exist_ok=True)
            path = os.path.join(prefix, path)
        pickle.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, name, prefix):
        """Load the pickled ElfiModel.

        Assumes there exists a file "name.pkl" in the current directory.

        Parameters
        ----------
        name : str
            Name of the model file to load (without the .pkl extension).
        prefix : str
            Path to directory where the model file is located, optional.

        Returns
        -------
        ElfiModel

        """
        path = name + '.pkl'
        if prefix is not None:
            path = os.path.join(prefix, path)
        return pickle.load(open(path, "rb"))

    def __getitem__(self, node_name):
        """Return a new reference object for a node in the model.

        Parameters
        ----------
        node_name : str

        """
        return self.get_reference(node_name)


class InstructionsMapper:
    @property
    def state(self):
        raise NotImplementedError()

    @property
    def uses_meta(self):
        return self.state['attr_dict'].get('_uses_meta', False)

    @uses_meta.setter
    def uses_meta(self, val):
        self.state['attr_dict']['_uses_meta'] = val


class NodeReference(InstructionsMapper):
    """A base class for node objects in the model.

    A user of ELFI will typically use, e.g. `elfi.Prior` or `elfi.Simulator` to create
    state dictionaries for nodes.

    Each node has a state dictionary that describes how the node ultimately produces its
    output (see module documentation for more details). The state is stored in the
    `ElfiModel` so that serializing the model is straightforward. `NodeReference` and its
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
        """Initialize a NodeReference.

        Parameters
        ----------
        parents : variable, optional
        name : string, optional
            If name ends in an asterisk '*' character, the asterisk will be replaced with
            a random string and the name is ensured to be unique within the model.
        state : dict, optional
        model : elfi.ElfiModel, optional

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
            model = get_default_model()

        return model

    @property
    def parents(self):
        """Get all positional parent nodes (inputs) of this node.

        Returns
        -------
        parents : list
            List of positional parents

        """
        return [self.model[p] for p in self.model.get_parents(self.name)]

    @classmethod
    def reference(cls, name, model):
        """Construct a reference for an existing node in the model.

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
        """Initialize all internal variables of the instance.

        Parameters
        ----------
        name : name of the node in the model
        model : ElfiModel

        """
        self.name = name
        self.model = model

    def generate(self, batch_size=1, with_values=None):
        """Generate output from this node.

        Useful for testing.

        Parameters
        ----------
        batch_size : int, optional
        with_values : dict, optional

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
        except BaseException:
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
        """Magic method that tries to infer the name from the code.

        Does not work in interactive python shell.
        """
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
        while re.match(r'\s*super\(', info.code_context[0]):
            frame = frame.f_back
            info = inspect.getframeinfo(frame, 1)

        # Match simple direct assignment with the class name, no commas or semicolons
        # Also do not accept a name starting with an underscore
        rex = r'\s*([^\W_][\w]*)\s*=\s*\w?[\w\.]*{}\('.format(self.__class__.__name__)
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
            if not model.has_node(name):
                break
        return name

    @property
    def state(self):
        """Return the state dictionary of the node."""
        if self.model is None:
            raise ValueError('{} {} is not initialized'.format(self.__class__.__name__, self.name))
        return self.model.get_node(self.name)

    def __getitem__(self, item):
        """Get item from the state dict of the node."""
        return self.state[item]

    def __setitem__(self, item, value):
        """Set item into the state dict of the node."""
        self.state[item] = value

    def __repr__(self):
        """Return a representation comprised of the names of the class and the node."""
        return "{}(name='{}')".format(self.__class__.__name__, self.name)

    def __str__(self):
        """Return the name of the node."""
        return self.name


class StochasticMixin(NodeReference):
    """Define the inheriting node as stochastic.

    Operations of stochastic nodes will receive a `random_state` keyword argument.
    """

    def __init__(self, *parents, state, **kwargs):
        # Flag that this node is stochastic
        state['_stochastic'] = True
        super(StochasticMixin, self).__init__(*parents, state=state, **kwargs)


class ObservableMixin(NodeReference):
    """Define the inheriting node as observable.

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
        """Initialize a node holding a constant value.

        Parameters
        ----------
        value
            The constant value of the node.

        """
        state = dict(_output=value)
        super(Constant, self).__init__(state=state, **kwargs)


class Operation(NodeReference):
    """A generic deterministic operation node."""

    def __init__(self, fn, *parents, **kwargs):
        """Initialize a node that performs an operation.

        Parameters
        ----------
        fn : callable
            The operation of the node.

        """
        state = dict(_operation=fn)
        super(Operation, self).__init__(*parents, state=state, **kwargs)


class RandomVariable(StochasticMixin, NodeReference):
    """A node that draws values from a random distribution."""

    def __init__(self, distribution, *params, size=None, **kwargs):
        """Initialize a node that represents a random variable.

        Parameters
        ----------
        distribution : str or scipy-like distribution object
        params : params of the distribution
        size : int, tuple or None, optional
            Output size of a single random draw.

        """
        state = dict(distribution=distribution, size=size, _uses_batch_size=True)
        state['_operation'] = self.compile_operation(state)
        super(RandomVariable, self).__init__(*params, state=state, **kwargs)

    @staticmethod
    def compile_operation(state):
        """Compile a callable operation that samples the associated distribution.

        Parameters
        ----------
        state : dict

        """
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
            raise ValueError("Distribution {} " "must implement a rvs method".format(distribution))

        op = partial(rvs_from_distribution, distribution=distribution, size=size)
        return op

    @property
    def distribution(self):
        """Return the distribution object."""
        distribution = self.state['attr_dict']['distribution']
        if isinstance(distribution, str):
            distribution = scipy_from_str(distribution)
        return distribution

    @property
    def size(self):
        """Return the size of the output from the distribution."""
        return self['size']

    def __repr__(self):
        """Return a string representation of the node."""
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
    """A parameter node of an ELFI graph."""

    def __init__(self, distribution, *params, size=None, **kwargs):
        """Initialize a Prior.

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
        self['attr_dict']['_parameter'] = True


class Simulator(StochasticMixin, ObservableMixin, NodeReference):
    """A simulator node of an ELFI graph.

    Simulator nodes are stochastic and may have observed data in the model.
    """

    def __init__(self, fn, *params, **kwargs):
        """Initialize a Simulator.

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
    """A summary node of an ELFI graph.

    Summary nodes are deterministic operations associated with the observed data. if their
    parents hold observed data it will be automatically transformed.
    """

    def __init__(self, fn, *parents, **kwargs):
        """Initialize a Summary.

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
    """A discrepancy node of an ELFI graph.

    This class provides a convenience node for custom distance operations.
    """

    def __init__(self, discrepancy, *parents, **kwargs):
        """Initialize a Discrepancy.

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
    """A convenience class for the discrepancy node."""

    def __init__(self, distance, *summaries, **kwargs):
        """Initialize a distance node of an ELFI graph.

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
        *summaries
            Summary nodes of the model.
        **kwargs
            Additional parameters may be required depending on the chosen distance.
            See the scipy documentation. (The support is not exhaustive.)
            ELFI-related kwargs are passed on to elfi.Discrepancy.

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

        Scipy distances:
        https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.spatial.distance.cdist.html  # noqa

        See Also
        --------
        elfi.Discrepancy : A general discrepancy node

        """
        if not summaries:
            raise ValueError("This node requires that at least one parent is specified.")

        if isinstance(distance, str):
            cdist_kwargs = dict(metric=distance)
            if distance == 'wminkowski' and 'w' not in kwargs.keys():
                raise ValueError('Parameter w must be specified for distance=wminkowski.')
            elif distance == 'seuclidean' and 'V' not in kwargs.keys():
                raise ValueError('Parameter V must be specified for distance=seuclidean.')
            elif distance == 'mahalanobis' and 'VI' not in kwargs.keys():
                raise ValueError('Parameter VI must be specified for distance=mahalanobis.')

            # extract appropriate keyword arguments (depends on distance, not exhaustive!)
            for key in ['p', 'w', 'V', 'VI']:
                if key in kwargs.keys():
                    cdist_kwargs[key] = kwargs.pop(key)

            dist_fn = partial(scipy.spatial.distance.cdist, **cdist_kwargs)
        else:
            dist_fn = distance

        discrepancy = partial(distance_as_discrepancy, dist_fn)
        super(Distance, self).__init__(discrepancy, *summaries, **kwargs)
        # Store the original passed distance
        self.state['distance'] = distance


class AdaptiveDistance(Discrepancy):
    """Euclidean (2-norm) distance calculation with adaptive scale.

    Summary statistics are normalised to vary on similar scales.

    References
    ----------
    Prangle D (2017). Adapting the ABC Distance Function. Bayesian
    Analysis 12(1):289-309, 2017.
    https://projecteuclid.org/euclid.ba/1460641065

    """

    def __init__(self, *summaries, **kwargs):
        """Initialize an AdaptiveDistance.

        Parameters
        ----------
        *summaries
            Summary nodes of the model.
        **kwargs

        Notes
        -----
        Your summaries need to be scalars or vectors for this method to
        work. The summaries will be first stacked to a single 2D array
        with the simulated summaries in the rows for every simulation
        and the distances are taken row wise against the corresponding
        observed summary vector.

        """
        if not summaries:
            raise ValueError("This node requires that at least one parent is specified.")

        discrepancy = partial(distance_as_discrepancy, self.nested_distance)
        super(AdaptiveDistance, self).__init__(discrepancy, *summaries, **kwargs)

        distance = partial(scipy.spatial.distance.cdist, metric='euclidean')
        self.state['attr_dict']['distance'] = distance
        self.init_state()

    def init_state(self):
        """Initialise adaptive distance state."""
        self.state['w'] = [None]
        dist_fn = partial(self.state['attr_dict']['distance'], w=None)
        self.state['distance_functions'] = [dist_fn]
        self.state['store'] = 3 * [None]
        self.init_adaptation_round()

    def init_adaptation_round(self):
        """Initialise data stores to start a new adaptation round."""
        if 'store' not in self.state:
            self.init_state()
        self.state['store'][0] = 0
        self.state['store'][1] = 0
        self.state['store'][2] = 0

    def add_data(self, *data):
        """Add summaries data to update estimated standard deviation.

        Parameters
        ----------
        *data
            Summary nodes output data.

        Notes
        -----
        Standard deviation is computed with Welford's online algorithm.

        """
        data = np.column_stack(data)

        self.state['store'][0] += len(data)
        delta_1 = data - self.state['store'][1]
        self.state['store'][1] += np.sum(delta_1, axis=0) / self.state['store'][0]
        delta_2 = data - self.state['store'][1]
        self.state['store'][2] += np.sum(delta_1 * delta_2, axis=0)

        self.state['scale'] = np.sqrt(self.state['store'][2]/self.state['store'][0])

    def update_distance(self):
        """Update distance based on accumulated summaries data."""
        weis = 1/self.state['scale']
        self.state['w'].append(weis)
        self.init_adaptation_round()
        dist_fn = partial(self.state['attr_dict']['distance'], w=weis**2)
        self.state['distance_functions'].append(dist_fn)

    def nested_distance(self, u, v):
        """Compute distance between simulated and observed summaries.

        Parameters
        ----------
        u : ndarray
            2D array with M x (num summaries) observations
        v : ndarray
            2D array with 1 x (num summaries) observations

        Returns
        -------
        ndarray
            2D array with M x (num distance functions) distances

        """
        return np.column_stack([d(u, v) for d in self.state['distance_functions']])
