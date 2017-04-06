import logging

import numpy as np

import elfi.client
from elfi.store import OutputPool
from elfi.bo.gpy_regression import GPyRegression
from elfi.bo.acquisition import LCBSC
from elfi.bo.utils import stochastic_optimization
from elfi.methods.posteriors import BolfiPosterior
from elfi.model.elfi_model import NodeReference, ElfiModel

logger = logging.getLogger(__name__)

__all__ = ['Rejection', 'BayesianOptimization', 'BOLFI']


"""

Implementing a new inference method
-----------------------------------

You can implement your own algorithm by subclassing the `InferenceMethod` class. The
methods that must be implemented raise `NotImplementedError`. In addition, you will
probably also want to override the `__init__` method. It can be useful to read through
`Rejection` and/or `BayesianOptimization` class implementations below to get you going. The
reason for the imposed structure in `InferenceMethod` is to encourage a design where one
can advance the inference iteratively, that is, to stop at any point, check the current
state and to be able to continue. This makes it possible to effectively tune the inference
as there are usually many moving parts, such as summary statistic choices or deciding the
best discrepancy function.

ELFI operates through batches. A batch is an indexed collection of one or more successive
outputs from the generative model (`ElfiModel`). The rule of thumb is that it should take
a significant amount of time to compute a batch. This makes it worthwhile to send it
over the network to a remote worker to be computed. A batch also needs to fit in to
memory.

ELFI guarantees that computing a batch with the same index will always produce the same
output given the same model and `ComputationContext` object. The `ComputationContext`
object holds the batch size, seed for the PRNG, and a pool of precomputed batches of nodes
and the observed values of the nodes.

When a new `InferenceMethod` is constructed, it will make a copy of the user provided
`ElfiModel` and make a new `ComputationContext` object for it. The user's model will stay
intact and the algorithm is free to modify it's copy as it needs to.


### Implementing the `__init__` method

You will need to call the `InferenceMethod.__init__` with a list of outputs, e.g. names of
nodes that you need the data for in each batch. For example, the rejection algorithm needs
the parameters and the discrepancy node output.

The first parameter to your `__init__` can be either the ElfiModel object or directly a
"target" node, e.g. discrepancy in rejection sampling. Assuming your `__init__` takes an
optional discrepancy parameter, you can detect which one was passed by using
`_resolve_model` method:

```
def __init__(model, discrepancy, ...):
    model, discrepancy = self._resolve_model(model, discrepancy)
```

In case you need multiple target nodes, you will need to write your own resolver.


### Explanations for some members of `InferenceMethod`

- objective : dict
    Holds the data for the algorithm to internally determine how many batches are still
    needed. In practice this information is used to determine the `_n_total_batches`
    together with the `state`.

- state : dict
    Stores any temporal data related to achieving the objective. Allows one to
    iteratively progress the inference and compute the `_n_total_batches` together with
    `state`.


### Good to know

#### `BatchHandler`

`InferenceMethod` class instantiates a `elfi.client.BatchHandler` helper class for you and
assigns it to `self.batches`. This object is in essence a wrapper to the `Client`
interface making it easier to work with batches that are in computation. Some of the
duties of `BatchHandler` is to keep track of the current batch_index and of the status of
the batches that have been submitted. You may however may not need to interact with it
directly.

#### `OutputPool`

`elfi.store.OutputPool` serves a dual purpose:
1. It stores the computed outputs of selected nodes
2. It provides those outputs when a batch is recomputed saving the need to recompute them.

If you want to provide values for outputs of certain nodes coming from outside the
generative model, you can store them to the `OutputPool` and they will replace the value
that would have otherwise been generated. This is used e.g. in `BOLFI` where values from
the acquisition function replace values coming from the prior in the Bayesian optimization
phase.

"""


class InferenceMethod(object):
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
        self.batch_size = batch_size

        # Prepare the computation_context
        context = model.computation_context.copy()
        if seed is not None:
            context.seed = seed
        context.batch_size = self.batch_size
        context.pool = pool

        self.model.computation_context = context

        self.client = elfi.client.get()
        self.batches = elfi.client.BatchHandler(self.model, outputs=outputs, client=self.client)

        self.max_parallel_batches = max_parallel_batches or self.client.num_cores

        # State and objective should contain all information needed to continue the
        # inference after an iteration.
        self.state = dict()
        self.objective = dict()

    @property
    def pool(self):
        return self.model.computation_context.pool

    @property
    def parameters(self):
        return self.model.parameters

    def init_inference(self, *args, **kwargs):
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

    def _get_batches_total(self):
        """ELFI calls this method to know how many batches should be submitted in total.
        This is allowed to change during the inference. ELFI will not submit at any given
        time more batches than allowed by `max_parallel_batches` which defaults to the
        number of cores in total.

        The inference loop will end when this number is reached and all batches currently
        in computation are finished.

        Returns
        -------
        batches_total : int

        """
        raise NotImplementedError

    def _prepare_new_batch(self, batch_index):
        """ELFI calls this method before submitting a new batch with an increasing index
        `batch_index`. This is an optional method to override. Use this if you have a need
        do do preparations, e.g. in Bayesian optimization algorithm, the next acquisition
        points would be acquired here.

        If you need provide values for certain nodes, you can do so by instantiating a
        pool for them in `__init__`. See e.g. BayesianOptimization for an example.

        Parameters
        ----------
        batch_index : int
            next batch_index to be submitted

        Returns
        -------
        None

        """
        pass

    def infer(self, *args, **kwargs):
        """Init the inference and start the iterate loop until the inference is finished.

        Returns
        -------
        result : dict
        """

        self.init_inference(*args, **kwargs)

        while not self.finished:
            self.iterate()

        return self.extract_result()

    def iterate(self):
        """Forward the inference one iteration. One iteration consists of processing the
        the result of the next batch in succession.

        If the next batch is ready, it will be processed immediately and no new batches
        are submitted.

        If the next batch is not ready, new batches will be submitted up to the
        `_n_total_batches` or `max_parallel_batches` or until the next batch is complete.

        If there are no more submissions to do and the next batch has still not finished,
        the method will wait for it's result.

        Returns
        -------
        None

        """

        # Submit new batches if allowed
        while self._allow_submit:
            batch_index = self.batches.next_index
            self._prepare_new_batch(batch_index)
            self.batches.submit()

        # Handle the next batch in succession
        batch, batch_index = self.batches.wait_next()
        self._update(batch, batch_index)

    @property
    def finished(self):
        return len(self.batches.pending_indices) == 0 and not self._has_batches_to_submit

    @property
    def _allow_submit(self):
        return self.max_parallel_batches > len(self.batches.pending_indices) and \
               self._has_batches_to_submit and \
               (not self.batches.has_ready)

    @property
    def _has_batches_to_submit(self):
        return self._get_batches_total() > self.batches.total

    def _to_array(self, batch, outputs=None):
        """Helper method to turn batches into numpy array
        
        Parameters
        ----------
        batch : dict or list
           Batch or list of batches
        outputs : list, optional
           Name of outputs to include in the array. Default is the `self.outputs`

        Returns
        -------
        np.array
            2d, where columns are batch outputs
        
        """

        if not batch:
            return []
        if not isinstance(batch, list):
            batch = [batch]
        outputs = outputs or self.outputs

        rows = []
        for batch_ in batch:
            rows.append(np.column_stack([batch_[output] for output in outputs]))

        return np.vstack(rows)

    @staticmethod
    def _resolve_model(model, target, default_reference_class=NodeReference):
        # TODO: extract the default_reference_class from model

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


class Rejection(InferenceMethod):
    """Parallel ABC rejection sampler.

    For a description of the rejection sampler and a general introduction to ABC, see e.g.
    Lintusaari et al. 2016.

    References
    ----------
    Lintusaari J, Gutmann M U, Dutta R, Kaski S, Corander J (2016). Fundamentals and
    Recent Developments in Approximate Bayesian Computation. Systematic Biology.
    http://dx.doi.org/10.1093/sysbio/syw077.
    """

    def __init__(self, model, discrepancy=None, **kwargs):
        """

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy : str or NodeReference
            Only needed if model is an ElfiModel
        kwargs:
            See InferenceMethod
        """

        model, self.discrepancy = self._resolve_model(model, discrepancy)
        outputs = model.parameters + [self.discrepancy]

        super(Rejection, self).__init__(model, outputs, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Parameters
        ----------
        n_samples : int
            Number of samples to generate from the (approximate) posterior

        Returns
        -------
        A dictionary with at least the following items:
        samples : list of np.arrays
            Samples from the posterior distribution of each parameter.
        """

        return self.infer(*args, **kwargs)

    def init_inference(self, n_samples, p=None, threshold=None):
        if p is None and threshold is None:
            p = .01
        self.state = dict(samples=None)
        self.objective = dict(n_samples=n_samples, p=p, threshold=threshold)
        # Reset the inference
        self.batches.clear()

    def _update(self, batch, batch_index):
        if self.state['samples'] is None:
            # Lazy initialization of the outputs dict
            self._init_state_samples(batch)
        self._merge_batch(batch)

    def extract_result(self):
        """Extracts the result from the current state"""
        if self.state['samples'] is None:
            raise ValueError('Nothing to extract')

        samples = self.state['samples']
        n_sim = self.batches.num_ready * self.batch_size
        n_samples = self.objective['n_samples']
        threshold = samples[self.discrepancy][n_samples - 1]
        accept_rate = n_samples / n_sim

        # Take out the correct number of samples
        for k, v in samples.items():
            samples[k] = v[:n_samples]

        result = dict(samples=samples,
                      n_sim=n_sim,
                      accept_rate=accept_rate,
                      threshold=threshold)

        return result

    def _get_batches_total(self):
        ks = ['p', 'threshold', 'n_sim', 'n_samples']
        p, t, n_sim, n_samples = [self.objective.get(k) for k in ks]

        total = 0
        if p:
            total = n_samples/(p*self.batch_size)
        elif n_sim:
            total = n_sim/self.batch_size
        elif t:
            n_current = 0
            if self.state['samples']:
                # noinspection PyTypeChecker
                n_current = np.sum(self.state['samples'][self.discrepancy] <= t)
            # TODO: make a smarter rule based on the average new samples / batch
            total = 2**64-1 if n_current < n_samples else 0

        return int(np.ceil(total))

    def _init_state_samples(self, batch):
        # Initialize the outputs dict based on the received batch
        samples = {}
        for node in self.outputs:
            shape = (self.objective['n_samples'] + self.batch_size,) \
                    + batch[node].shape[1:]
            samples[node] = np.ones(shape) * np.inf
        self.state['samples'] = samples

    def _merge_batch(self, batch):
        # TODO: add index vector so that you can recover the original order, also useful
        #       for async

        samples = self.state['samples']

        # Put the acquired samples to the end
        for node, v in samples.items():
            v[self.objective['n_samples']:] = batch[node]

        # Sort the smallest to the beginning
        sort_mask = np.argsort(samples[self.discrepancy], axis=0).ravel()
        for k, v in samples.items():
            v[:] = v[sort_mask]


class BayesianOptimization(InferenceMethod):
    """Bayesian Optimization of an unknown target function."""

    def __init__(self, model, target=None, batch_size=1, n_acq=150, initial_evidence=10,
                 update_interval=10, bounds=None, target_model=None,
                 acquisition_method=None, pool=None, **kwargs):
        """
        Parameters
        ----------
        model : ElfiModel or NodeReference
        target : str or NodeReference
            Only needed if model is an ElfiModel
        target_model : GPyRegression, optional
        acquisition_method : Acquisition, optional
        bounds : list
            The region where to estimate the posterior for each parameter in
            model.parameters.
            `[(lower, upper), ... ]`
        initial_evidence : int, dict
            Number of initial evidence or a precomputed batch dict containing parameter 
            and discrepancy values
        n_evidence : int
            The total number of evidence to acquire for the target_model regression
        update_interval : int
            How often to update the GP hyperparameters of the target_model
        exploration_rate : float
            Exploration rate of the acquisition method
        """

        model, self.target = self._resolve_model(model, target)
        outputs = model.parameters + [self.target]

        # We will need a pool
        pool = pool or OutputPool()

        super(BayesianOptimization, self).__init__(model, outputs=outputs,
                                                   batch_size=batch_size, pool=pool,
                                                   **kwargs)

        # Init the pool
        for param in self.outputs:
            if param not in self.pool:
                self.pool.add_store(param, {})

        target_model = \
            target_model or GPyRegression(len(self.model.parameters), bounds=bounds)

        if not isinstance(initial_evidence, int):
            # Add precomputed batch data
            params = self._to_array(initial_evidence, self.parameters)
            target_model.update(params, initial_evidence[self.target])
            initial_evidence = len(params)

        # TODO: check if this can be removed
        if initial_evidence % self.batch_size != 0:
            raise ValueError('Initial evidence must be divisible by the batch size')

        self.acquisition_method = acquisition_method or LCBSC(target_model)

        self.target_model = target_model
        self.n_initial_evidence = initial_evidence
        self.n_acq = n_acq
        self.update_interval = update_interval

    def init_inference(self, n_acq=None):
        """You can continue BO by giving a larger n_acq"""
        self.state = self.state or dict(last_update=0)

        if n_acq and self.n_acq > n_acq:
            raise ValueError('New n_acq must be greater than the earlier')

        self.n_acq = n_acq or self.n_acq

    def extract_result(self):
        param, min_value = stochastic_optimization(self.target_model.predict_mean,
                                                   self.target_model.bounds)
        # TODO: use this such that the result from this method is not arrays
        # param_hat = dict(zip(self.parameters, param))

        param_hat = {}
        for i, p in enumerate(self.model.parameters):
            # Preserve as array
            param_hat[p] = param[i:i + 1]

        return dict(samples=param_hat)

    def _update(self, batch, batch_index):
        """Update the GP regression model of the target node.
        """
        params = self._to_array(batch, self.parameters)
        self._report_batch(batch_index, params, batch[self.target])

        optimize = self._should_optimize()
        self.target_model.update(params, batch[self.target], optimize)

        if optimize:
            self.state['last_update'] = self.target_model.n_evidence

    def _get_batches_total(self):
        return int(np.ceil((self.n_acq + self.n_initial_evidence) / self.batch_size))

    def _prepare_new_batch(self, batch_index):
        if self._n_submitted_evidence < self.n_initial_evidence:
            return

        pending_params = self._get_pending_params()
        t = self.batches.total - int(self.n_initial_evidence/self.batch_size)
        new_param = self.acquisition_method.acquire(self.batch_size, pending_params, t)

        # Add the next evaluation location to the pool
        # TODO: make to_batch method
        batch = {p: new_param[:,i:i+1] for i, p in enumerate(self.parameters)}
        self.pool.add_batch(batch_index, batch)

    @property
    def _n_submitted_evidence(self):
        return self.batches.total*self.batch_size

    @property
    def _allow_submit(self):
        # Do not start acquisition unless all of the initial evidence is ready
        prevent = self._n_submitted_evidence >= self.n_initial_evidence and \
                  self.target_model.n_evidence < self.n_initial_evidence
        return super(BayesianOptimization, self)._allow_submit and not prevent

    def _should_optimize(self):
        current = self.target_model.n_evidence + self.batch_size
        next_update = self.state['last_update'] + self.update_interval
        return current >= self.n_initial_evidence and current >= next_update

    def _get_pending_params(self):
        # Prepare pending locations for the acquisition
        pending_batches = [self.pool.get_batch(i, self.parameters) for i in
                           self.batches.pending_indices]
        return self._to_array(pending_batches, self.parameters)

    def _report_batch(self, batch_index, params, distances):
        str = "Received batch {}:\n".format(batch_index)
        fill = 6 * ' '
        for i in range(self.batch_size):
            str += "{}{} at {}\n".format(fill, distances[i].item(), params[i])
        logger.debug(str)


class BOLFI(InferenceMethod):
    """Bayesian Optimization for Likelihood-Free Inference (BOLFI).

    Approximates the discrepancy function by a stochastic regression model.
    Discrepancy model is fit by sampling the discrepancy function at points decided by
    the acquisition function.

    The implementation follows that of Gutmann & Corander, 2016.

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html

    """

    def __init__(self, model, batch_size=1, discrepancy=None, bounds=None, **kwargs):
        """
        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy : str or NodeReference
            Only needed if model is an ElfiModel
        discrepancy_model : GPRegression, optional
        acquisition_method : Acquisition, optional
        bounds : dict
            The region where to estimate the posterior for each parameter;
            `dict(param0: (lower, upper), param2: ... )`
        initial_evidence : int, dict
            Number of initial evidence or a precomputed dict containing parameter and
            discrepancy values
        n_evidence : int
            The total number of evidence to acquire for the discrepancy_model regression
        update_interval : int
            How often to update the GP hyperparameters of the discrepancy_model
        exploration_rate : float
            Exploration rate of the acquisition method
        """

    def get_posterior(self, threshold):
        """Returns the posterior.

        Parameters
        ----------
        threshold: float
            discrepancy threshold for creating the posterior

        Returns
        -------
        BolfiPosterior object
        """
        return BolfiPosterior(self.discrepancy_model, threshold)

