import logging
from operator import itemgetter


import numpy as np

import elfi.client
from elfi.bo.gpy_model import GPyRegression
from elfi.bo.acquisition import BolfiAcquisition, UniformAcquisition, LCB, LCBSC
from elfi.bo.utils import stochastic_optimization
from elfi.methods.posteriors import BolfiPosterior
from elfi.model.elfi_model import NodeReference, ElfiModel, Discrepancy

logger = logging.getLogger(__name__)

__all__ = ['Rejection', 'BayesianOptimization', 'BOLFI']


"""Implementations of simulator based inference algorithms.


Implementing a new algorithm
----------------------------

objective : dict
    Holds the minimum data required to determine when the inference is finished
state : dict
    Holds any temporal data of the state related to achieving the objective. This will be
    reset when inference is restarted so do not store here anything that needs to be
    preserved across inferences with the same inference method instance.

"""


class InferenceMethod(object):
    """
    """

    def __init__(self, model, outputs, batch_size=1000, seed=None, pool=None,
                 max_parallel_batches=None):
        """

        Parameters
        ----------
        model : ElfiModel or NodeReference
        outputs : iterable
            Contains the node names
        batch_size : int
        seed : int

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

        self.max_parallel_batches = max_parallel_batches or self.client.num_cores + 1

        self.state = None
        self.objective = None

    def init_inference(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, batch, batch_index):
        raise NotImplementedError

    def prepare_new_batch(self, batch_index):

        pass

    def extract_result(self):
        raise NotImplementedError

    @property
    def estimated_total_batches(self):
        """Can change during execution if needed"""
        raise NotImplementedError

    def infer(self, *args, **kwargs):
        self.init_inference(*args, **kwargs)

        while not self.finished:
            self.iterate()

        return self.extract_result()

    def iterate(self):
        # Submit new batches
        while self.allow_submit:
            batch_index = self.batches.next_index
            self.prepare_new_batch(batch_index)
            self.batches.submit()

        # Handle all received batches
        while True:
            batch, batch_index = self.batches.wait_next()
            self.update(batch, batch_index)
            if not self.batches.has_ready: break

    @property
    def finished(self):
        return len(self.batches.pending) == 0 and not self.has_batches_to_submit

    @property
    def allow_submit(self):
        return self.max_parallel_batches > len(self.batches.pending) and\
               (not self.batches.has_ready) and\
               self.has_batches_to_submit

    @property
    def has_batches_to_submit(self):
        return self.estimated_total_batches > self.batches.total

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

    For introduction to ABC, see e.g. Lintusaari et al. 2016.

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
            Number of samples from the posterior

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
        self.state = dict(outputs=None)
        self.objective = dict(n_samples=n_samples, p=p, threshold=threshold)
        self.batches.reset()

    def update(self, batch, batch_index):
        # Initialize the outputs dict
        if self.state['outputs'] is None:
            self._init_outputs(batch)
        self._merge_batch(batch)

    def extract_result(self):
        """Extracts the result from the current state"""
        if self.state['outputs'] is None:
            raise ValueError('Nothing to extract')

        outputs = self.state['outputs']
        n_sim = self.batches.num_ready * self.batch_size
        n_samples = self.objective['n_samples']
        threshold = outputs[self.discrepancy][n_samples - 1]
        accept_rate = n_samples / n_sim

        # Take out the correct number of samples
        for k, v in outputs.items():
            outputs[k] = v[:n_samples]

        result = dict(samples=outputs,
                      n_sim=n_sim,
                      accept_rate=accept_rate,
                      threshold=threshold)

        return result

    @property
    def estimated_total_batches(self):
        ks = ['p', 'threshold', 'n_sim', 'n_samples']
        p, t, n_sim, n_samples = [self.objective.get(k) for k in ks]

        total = 0
        if p:
            total = n_samples/(p*self.batch_size)
        elif n_sim:
            total = n_sim/self.batch_size
        elif t:
            n_current = 0
            if self.state['outputs']:
                # noinspection PyTypeChecker
                n_current = np.sum(self.state['outputs'][self.discrepancy] <= t)
            # TODO: make a smarter rule based on the average new samples / batch
            total = 2**64-1 if n_current < n_samples else 0

        return int(np.ceil(total))

    def _init_outputs(self, batch):
        # Initialize the outputs dict based on the received batch
        outputs = {}
        for output in self.outputs:
            shape = (self.objective['n_samples'] + self.batch_size,) \
                    + batch[output].shape[1:]
            outputs[output] = np.ones(shape) * np.inf
        self.state['outputs'] = outputs

    def _merge_batch(self, batch):
        # TODO: add index vector so that you can recover the original order, also useful
        #       for async

        outputs = self.state['outputs']

        # Put the acquired outputs to the end
        for k, v in outputs.items():
            v[self.objective['n_samples']:] = batch[k]

        # Sort the smallest to the beginning
        sort_mask = np.argsort(outputs[self.discrepancy], axis=0).ravel()
        for k, v in outputs.items():
            v[:] = v[sort_mask]


class BayesianOptimization(InferenceMethod):
    """Bayesian Optimization of an unknown target function."""

    def __init__(self, model, target=None, batch_size=1, n_acq=150,
                 initial_evidence=10, update_interval=10, exploration_rate=10,
                 bounds=None, target_model=None, acquisition_method=None, **kwargs):
        """
        Parameters
        ----------
        model : ElfiModel or NodeReference
        target : str or NodeReference
            Only needed if model is an ElfiModel
        target_model : GPRegression, optional
        acquisition_method : Acquisition, optional
        bounds : dict
            The region where to estimate the posterior for each parameter;
            `dict(param0: (lower, upper), param2: ... )`
        initial_evidence : int, dict
            Number of initial evidence or a precomputed dict containing parameter and
            discrepancy values
        n_evidence : int
            The total number of evidence to acquire for the target_model regression
        update_interval : int
            How often to update the GP hyperparameters of the target_model
        exploration_rate : float
            Exploration rate of the acquisition method
        """

        model, self.target = self._resolve_model(model, target)
        outputs = model.parameters + [self.target]

        super(BayesianOptimization, self).__init__(model, outputs=outputs,
                                                   batch_size=batch_size, **kwargs)

        supply = {}
        for param in self.model.parameters:
            self.model.computation_context.output_supply[param] = supply

        target_model = \
            target_model or GPyRegression(len(self.model.parameters), bounds=bounds)

        if not isinstance(initial_evidence, int):
            raise NotImplementedError('Initial evidence must be an integer')

        if initial_evidence > 0:
            self.init_acquisition = UniformAcquisition(target_model)
        else:
            self.init_acquisition = None

        self.acquisition_method = acquisition_method or \
                                  LCBSC(target_model, exploration_rate=exploration_rate)

        self.target_model = target_model
        self.n_initial_evidence = initial_evidence
        self.n_acq = n_acq
        self.update_interval = update_interval

    def init_inference(self, n_acq=None):
        self.n_acq = n_acq or self.n_acq
        self.state = dict(last_update=0)
        self.batches.reset()

    def update(self, batch, batch_index):
        """Update the GP regression model of the target node.
        """
        # TODO: do not use the output_supply
        context = self.model.computation_context
        pending_batches = context.output_supply[self.model.parameters[0]]
        pending_batches.pop(batch_index)

        current = self.target_model.n_observations + self.batch_size
        next_update = self.state['last_update'] + self.update_interval

        if current >= self.n_initial_evidence and current >= next_update:
            optimize = True
            self.state['last_update'] = current
        else: optimize = False

        params = np.atleast_2d([batch[param] for param in self.model.parameters])
        for i in range(self.batch_size):
            logger.debug("Observed batch {}: {} at {}".format(batch_index,
                                                              batch[self.target][i][0],
                                                              params[i]))

        # TODO: should target_model.update accept directly the batch?
        self.target_model.update(params, batch[self.target], optimize)

    def prepare_new_batch(self, batch_index):
        context = self.model.computation_context
        pending_batches = context.output_supply[self.model.parameters[0]]

        # TODO: take values from the pool
        pending_params = []
        for output in pending_batches.values():
            pending_params.append([output[param] for param in self.model.parameters])
        pending_params = np.vstack(pending_params) if pending_params else None

        t = self.batches.total
        if t >= self.n_initial_evidence:
            new_param = self.acquisition_method.acquire(1, pending_params, t)
        else:
            new_param = self.init_acquisition.acquire(1, pending_params, t)

        # Set the next evaluation location
        pending_batches[batch_index] = dict(zip(self.model.parameters, new_param[0]))

    def extract_result(self):
        param, min_value = stochastic_optimization(self.target_model.evaluate_mean,
                                                   self.target_model.bounds)
        result = {}
        for i, p in enumerate(self.model.parameters):
            # TODO: stochastic optimization should return a numpy array
            result[p] = np.atleast_1d([param[i]])

        return dict(samples=result)

    @property
    def estimated_total_batches(self):
        return self.n_acq + self.n_initial_evidence

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
        return BolfiPosterior(self.target_model, threshold)


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

