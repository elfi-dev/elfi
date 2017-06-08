Implementing a new method
=========================

You can implement a custom parameter inference method by subclassing the
``ParameterInference`` interface class which is found from the
``elfi.methods.parameter_inference`` module. Among the methods in the interface, those
that need to be implemented raise a ``NotImplementedError``. In addition, you probably
also want to override the ``__init__`` method. Let's create a skeleton for a new
method::

class ParameterInference(object):
    """
    """

    def __init__(self, model, outputs, batch_size=1000, seed=None, pool=None,
                 max_parallel_batches=None):
        """Construct the inference algorithm object.

        If you are implementing your own algorithm do not forget to call `super`.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        outputs : list
            Contains the node names for which the algorithm needs to receive the outputs
            in every batch.
        batch_size : int
        seed : int
            Seed for the data generation from the ElfiModel
        pool : OutputPool
            OutputPool both stores and provides precomputed values for batches.
        max_parallel_batches : int
            Maximum number of batches allowed to be in computation at the same time.
            Defaults to number of cores in the client


        """
        model = model.model if isinstance(model, NodeReference) else model

        if not model.parameters:
            raise ValueError('Model {} defines no parameters'.format(model))

        self.model = model.copy()
        self.outputs = outputs

        # Prepare the computation_context
        context = ComputationContext(
            seed=seed,
            batch_size=batch_size,
            observed=model.computation_context.observed,
            pool=pool
        )
        self.model.computation_context = context
        self.client = elfi.client.get_client()
        self.batches = elfi.client.BatchHandler(self.model, outputs=outputs, client=self.client)
        self.max_parallel_batches = max_parallel_batches or self.client.num_cores

        if self.max_parallel_batches <= 0:
            msg = 'Value for max_parallel_batches ({}) must be at least one.'.format(
                self.max_parallel_batches)
            if self.client.num_cores == 0:
                msg += ' Client has currently no workers available. Please make sure ' \
                       'the cluster has fully started or set the max_parallel_batches ' \
                       'parameter by hand.'
            raise ValueError(msg)

        # State and objective should contain all information needed to continue the
        # inference after an iteration.
        self.state = dict(n_sim=0, n_batches=0)
        self.objective = dict(n_batches=0)

    @property
    def pool(self):
        return self.model.computation_context.pool

    @property
    def seed(self):
        return self.model.computation_context.seed

    @property
    def parameters(self):
        return self.model.parameters

    @property
    def batch_size(self):
        return self.model.computation_context.batch_size

    def set_objective(self, *args, **kwargs):
        """This method is called when one wants to begin the inference. Set `self.state`
        and `self.objective` here for the inference.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def extract_result(self):
        """This method is called when one wants to receive the result from the inference.
        You should prepare the output here and return it.

        Returns
        -------
        result : dict
        """
        raise NotImplementedError

    def _update(self, batch, batch_index):
        """ELFI calls this method when a new batch has been computed and the state of
        the inference should be updated with it.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        Returns
        -------
        None
        """
        raise NotImplementedError

    def _prepare_new_batch(self, batch_index):
        """ELFI calls this method before submitting a new batch with an increasing index
        `batch_index`. This is an optional method to override. Use this if you have a need
        do do preparations, e.g. in Bayesian optimization algorithm, the next acquisition
        points would be acquired here.

        If you need provide values for certain nodes, you can do so by constructing a
        batch dictionary and returning it. See e.g. BayesianOptimization for an example.

        Parameters
        ----------
        batch_index : int
            next batch_index to be submitted

        Returns
        -------
        batch : dict or None

        """
        pass

    def plot_state(self, **kwargs):
        """

        Parameters
        ----------
        axes : matplotlib.axes.Axes (optional)
        figure : matplotlib.figure.Figure (optional)
        xlim
            x-axis limits
        ylim
            y-axis limits
        interactive : bool (default False)
            If true, uses IPython.display to update the cell figure
        close
            Close figure in the end of plotting. Used in the end of interactive mode.

        Returns
        -------
        None

        """
        raise NotImplementedError

    def infer(self, *args, vis=None, **kwargs):
        """Init the inference and start the iterate loop until the inference is finished.

        Returns
        -------
        result : Result
        """
        vis_opt = vis if isinstance(vis, dict) else {}

        self.set_objective(*args, **kwargs)

        while not self.finished:
            self.iterate()
            if vis:
                self.plot_state(interactive=True, **vis_opt)

        self.batches.cancel_pending()
        if vis:
            self.plot_state(close=True, **vis_opt)
        return self.extract_result()

    def iterate(self):
        """Forward the inference one iteration.

        This is a way to manually progress the inference. One iteration consists of
        waiting and processing the result of the next batch in succession and possibly
        submitting new batches.

        Notes
        -----
        If the next batch is ready, it will be processed immediately and no new batches
        are submitted.

        New batches are submitted only while waiting for the next one to complete. There
        will never be more batches submitted in parallel than the `max_parallel_batches`
        setting allows.

        Returns
        -------
        None

        """

        # Submit new batches if allowed
        while self._allow_submit:
            batch_index = self.batches.next_index
            batch = self._prepare_new_batch(batch_index)
            self.batches.submit(batch)

        # Handle the next batch in succession
        batch, batch_index = self.batches.wait_next()
        self._update(batch, batch_index)

    @property
    def finished(self):
        return self.objective['n_batches'] <= self.state['n_batches']

    @property
    def _allow_submit(self):
        return self.max_parallel_batches > self.batches.num_pending and \
               self._has_batches_to_submit and \
               (not self.batches.has_ready)

    @property
    def _has_batches_to_submit(self):
        return self.objective['n_batches'] > self.state['n_batches'] + self.batches.num_pending

    def _to_array(self, batches, outputs=None):
        """Helper method to turn batches into numpy array

        Parameters
        ----------
        batches : list or dict
           A list of batches or as single batch
        outputs : list, optional
           Name of outputs to include in the array. Default is the `self.outputs`

        Returns
        -------
        np.array
            2d, where columns are batch outputs

        """

        if not batches:
            return []
        if not isinstance(batches, list):
            batches = [batches]
        outputs = outputs or self.outputs

        rows = []
        for batch_ in batches:
            rows.append(np.column_stack([batch_[output] for output in outputs]))

        return np.vstack(rows)

    @staticmethod
    def _resolve_model(model, target, default_reference_class=NodeReference):
        # TODO: extract the default_reference_class from the model?

        if isinstance(model, ElfiModel) and target is None:
            raise NotImplementedError("Please specify the target node of the inference method")

        if isinstance(model, NodeReference):
            target = model
            model = target.model

        if isinstance(target, str):
            target = model[target]

        if not isinstance(target, default_reference_class):
            raise ValueError('Unknown target node class')

        return model, target.name

    @staticmethod
    def _ensure_outputs(outputs, required_outputs):
        outputs = outputs or []
        for out in required_outputs:
            if out not in outputs:
                outputs.append(out)
        return outputs

It can be useful to read
through ``Rejection``, ``SMC`` and/or ``BayesianOptimization`` class implementations below to
get you going. The reason for the imposed structure in ``ParameterInference`` is to encourage a
design where one can advance the inference iteratively, that is, to stop at any point,
check the current state and to be able to continue. This makes it possible to effectively
tune the inference as there are usually many moving parts, such as summary statistic
choices or deciding the best discrepancy function.

ELFI operates through batches. A batch is an indexed collection of one or more successive
outputs from the generative model (``ElfiModel``). The rule of thumb is that it should take
a significant amount of time to compute a batch. This ensures that it is worthwhile to
send a batch over the network to a remote worker to be computed. A batch also needs to fit
into memory.

ELFI guarantees that computing a batch with the same index will always produce the same
output given the same model and ``ComputationContext`` object. The ``ComputationContext``
object holds the batch size, seed for the PRNG, and a pool of precomputed batches of nodes
and the observed values of the nodes.

When a new ``ParameterInference`` is constructed, it will make a copy of the user provided
``ElfiModel`` and make a new ``ComputationContext`` object for it. The user's model will stay
intact and the algorithm is free to modify it's copy as it needs to.


Implementing the ``__init__`` method
------------------------------------

You will need to call the ``InferenceMethod.__init__`` with a list of outputs, e.g. names of
nodes that you need the data for in each batch. For example, the rejection algorithm needs
the parameters and the discrepancy node output.

The first parameter to your ``__init__`` can be either the ElfiModel object or directly a
"target" node, e.g. discrepancy in rejection sampling. Assuming your ``__init__`` takes an
optional discrepancy parameter, you can detect which one was passed by using
``_resolve_model`` method::

   def __init__(model, discrepancy, ...):
      model, discrepancy = self._resolve_model(model, discrepancy)

In case you need multiple target nodes, you will need to write your own resolver.


Explanations for some members of ``ParameterInference``
-------------------------------------------------------

- objective : dict
    Holds the data for the algorithm to internally determine how many batches are still
    needed. You must have a key ``n_batches`` here. This information is used to determine
    when the algorithm is finished.

- state : dict
    Stores any temporal data related to achieving the objective. Must include a key
    ``n_batches`` for determining when the inference is finished.


Good to know
------------

``BatchHandler``
................

``ParameterInference`` class instantiates a ``elfi.client.BatchHandler`` helper class for you and
assigns it to ``self.batches``. This object is in essence a wrapper to the ``Client``
interface making it easier to work with batches that are in computation. Some of the
duties of ``BatchHandler`` is to keep track of the current batch_index and of the status of
the batches that have been submitted. You may however may not need to interact with it
directly.

``OutputPool``
..............

``elfi.store.OutputPool`` serves a dual purpose:
1. It stores the computed outputs of selected nodes
2. It provides those outputs when a batch is recomputed saving the need to recompute them.

If you want to provide values for outputs of certain nodes from outside the generative
model, you can return then in ``prepare_new_batch`` method. They will be inserted into to
the ``OutputPool`` and will replace any value that would have otherwise been generated from
the node. This is used e.g. in ``BOLFI`` where values from the acquisition function replace
values coming from the prior in the Bayesian optimization phase.
