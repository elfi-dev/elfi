import logging

import numpy as np

import elfi.client
from elfi.bo.gpy_model import GPyRegression
from elfi.bo.acquisition import BolfiAcquisition, UniformAcquisition, LCB, LCBSC
from elfi.methods.posteriors import BolfiPosterior
from elfi.model.elfi_model import NodeReference, ElfiModel, Discrepancy

logger = logging.getLogger(__name__)

__all__ = ['Rejection', 'BOLFI']


"""Implementations of simulator based inference algorithms.
"""


class InferenceMethod(object):
    """
    """

    def __init__(self, model, outputs, batch_size=1000, seed=None, pool=None,
                 max_concurrent_batches=None):
        """

        Parameters
        ----------
        model : ElfiModel or NodeReference
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

        self.max_concurrent_batches = max_concurrent_batches or self.client.num_cores + 1

        self.state = None
        self.objective = None

    def init_inference(self, *args, **kwargs):
        raise NotImplementedError

    def update_state(self, batch, batch_index):
        raise NotImplementedError

    def prepare_new_batch(self, batch_index):
        pass

    def extract_result(self):
        raise NotImplementedError

    @property
    def estimated_num_batches(self):
        """Can change during execution if needed"""
        raise NotImplementedError

    def sample(self, n_samples, *args, **kwargs):
        """Run the sampler.

        Subsequent calls will reuse existing data without rerunning the
        simulator until necessary.

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

        self.init_inference(*args, n_samples=n_samples, **kwargs)

        while self.running:
            self.iterate()

        return self.extract_result()

    def iterate(self):

        if self.objective is None:
            raise ValueError("Objective is not initialized")
        if self.state is None:
            raise ValueError("State is not initialized")

        while self.allow_submit:
            batch_index = self.batches.next
            self.prepare_new_batch(batch_index)
            self.batches.submit()

        batch, batch_index = self.batches.wait_next()

        self.update_state(batch, batch_index)

    @property
    def running(self):
        return len(self.batches.pending) > 0 or self.estimated_num_batches > 0

    @property
    def allow_submit(self):
        if self.max_concurrent_batches <= len(self.batches.pending):
            return False
        elif self.batches.has_ready:
            return False
        else:
            return self.estimated_num_batches > 0

    @staticmethod
    def _resolve_target(model, target, default_reference_class=object):
        # TODO: extract the default_reference_class from model
        if target is None:
            target = model

        if isinstance(target, ElfiModel):
            raise NotImplementedError("Not implemented yet, please use NodeReference "
                                      "instances instead, e.g. model['d']")

        if not isinstance(target, NodeReference):
            raise ValueError('Unknown target node')

        return target.name, target.model


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
        seed : int
        batch_size : int
        discrepancy : str or NodeReference (optional)
            Name of the discrepancy node. Needed only if discrepancy node cannot be
            inferred from the model

        """

        self.discrepancy, model = self._resolve_target(model, discrepancy)
        outputs = model.parameters + [self.discrepancy]

        super(Rejection, self).__init__(model, outputs, **kwargs)

    def init_inference(self, n_samples, p=None, threshold=None):
        if p is None and threshold is None:
            p = .01
        self.state = dict(outputs=None)
        self.objective = dict(n_samples=n_samples, p=p, threshold=threshold)
        self.batches.reset()

    def update_state(self, batch, batch_index):
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

        result = dict(outputs=outputs,
                      n_sim=n_sim,
                      accept_rate=accept_rate,
                      threshold=threshold)

        return result

    @property
    def estimated_num_batches(self):
        # Get the conditions
        p = self.objective.get('p')
        nsim = self.objective.get('n_sim')
        t = self.objective.get('threshold')
        n_samples = self.objective['n_samples']

        n_needed = 0
        if p:
            n_needed = n_samples/(p*self.batch_size)
        elif nsim:
            n_needed = nsim/self.batch_size
        elif t:
            n_current = 0
            if self.state['outputs']:
                # noinspection PyTypeChecker
                n_current = np.sum(self.state['outputs'][self.discrepancy] <= t)
            # TODO: make a smarter rule based on the average new samples / batch
            n_needed = 2**64-1 if n_current < n_samples else 0

        n_needed = int(np.ceil(n_needed))
        return max(n_needed - self.batches.total, 0)

    def _init_outputs(self, batch):
        # Initialize the outputs dict
        n_samples = self.objective['n_samples']
        outputs = {}
        for k in self.outputs:
            v = batch[k]
            outputs[k] = np.ones((n_samples + self.batch_size,) + v.shape[1:]) * np.inf
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

    Parameters
    ----------
    model : ElfiModel or NodeReference
    seed : int
    batch_size : int
    discrepancy : str or NodeReference
    discrepancy_model : GPRegression
    acquisition : acquisition function object
    store : various (optional)
    async : bool
        Use asynchronous sampling (default: False).
    bounds : dict
        The region where to estimate the posterior for each parameter;
        `dict(parameter_name : (lower, upper) )`
    initial_evidence : int, dict
        Number of initial evidence or a precomputed dict containing parameter and
        discrepancy values
    n_total_evidence : int
        The total number of evidence to acquire for the discrepancy_model regression

    optimizer : string
        See GPyModel
    max_opt_iters : int
        See GPyModel
    """

    def __init__(self, model, seed=None, batch_size=1, max_parallel_acquisitions=10,
                 store=None, acquisition=None, discrepancy=None, discrepancy_model=None,
                 async=False, bounds=None, initial_evidence=None, n_total_evidence=None,
                 update_interval=100,
                 optimizer="scg", max_opt_iters=None, exploration_rate=10):



        super(BOLFI, self).__init__(model, batch_size=batch_size, seed=seed)

        self.discrepancy = self._resolve_target(model, discrepancy)

        self.discrepancy_model = discrepancy_model or \
                                 GPyRegression(len(self.model.parameters), bounds=bounds,
                                               optimizer=optimizer,
                                               max_opt_iters=max_opt_iters)

        self.max_parallel_acquisitions = max_parallel_acquisitions
        self.async = async

        init_acquisition = None
        # TODO: perhaps add a rule of thumb based on the number of dimensios
        if initial_evidence is None or isinstance(initial_evidence, int):
            n_initial = initial_evidence or max(self.batches.num_cores(), 10)
            init_acquisition = UniformAcquisition(self.discrepancy_model, n_initial)
            n_total_evidence -= n_initial
            self.n_initial = n_initial

        self.acquisition = acquisition or LCBSC(self.discrepancy_model, n_total_evidence,
                                              exploration_rate=exploration_rate)
        #BolfiAcquisition(self.discrepancy_model, n_samples=n_total_evidence)

        if init_acquisition:
            self.acquisition = init_acquisition + self.acquisition

        self.update_interval = update_interval

        self.compiled_net = \
            self.batches.compile(self.model.source_net,
                                 outputs=self.model.parameters + [self.discrepancy])

        supply = {}
        for param in self.model.parameters:
            self.model.computation_context.output_supply[param] = supply

    def infer(self, threshold=None):
        """Bolfi inference.

        Parameters
        ----------
        see get_posterior

        Returns
        -------
        see get_posterior
        """
        self.fit()
        return self.get_posterior(threshold)

    def fit(self):
        """Fits the GP model to the discrepancy random variable.
        """
        if self.async:
            logger.info("Using async mode")
        logger.info("Evaluating {:d} batches of size {:d}.".format(self.acquisition.batches_left,
                                                                   self.batch_size))

        # TODO: move batch index handling to client (client.submitted_indexes would return
        #       a list of indexes submitted)

        context = self.model.computation_context
        pending_batches = context.output_supply[self.model.parameters[0]]
        n_batches = 0
        last_update = 0

        while not self.acquisition.finished or self.batches.has_pending():
            n_new_batches, new_indexes = self._next_num_batches(
                self.batches.num_pending_batches(), n_batches)
            n_batches += n_new_batches

            # TODO: change acquisition format to match with output format
            locs = []
            for output in pending_batches.values():
                locs.append([output[param] for param in self.model.parameters])
            locs = np.vstack(locs) if locs else None

            t = n_batches - n_new_batches - self.batches.num_pending_batches()
            new_locs = self.acquisition.acquire(n_new_batches, locs, t)

            for i in range(n_new_batches):
                pending_batches[new_indexes[i]] = dict(zip(self.model.parameters, new_locs[i]))

            self.batches.submit_batches(new_indexes, self.compiled_net, context)

            batch_outputs, batch_index = self.batches.wait_next()
            pending_batches.pop(batch_index)

            location = np.array([batch_outputs[param] for param in self.model.parameters])

            for i in range(self.batch_size):
                logger.debug("Observed {} at {}".format(
                    batch_outputs[self.discrepancy][i], location))

            optimize = False
            if self.discrepancy_model.n_observations+self.batch_size >= last_update + self.update_interval:
                if self.n_initial <= self.discrepancy_model.n_observations+self.batch_size:
                    optimize = True
                    last_update = self.discrepancy_model.n_observations + self.batch_size

            self.discrepancy_model.update(np.atleast_2d(location),
                                          batch_outputs[self.discrepancy], optimize)

    def _next_num_batches(self, n_pending, current_batch_index):
        """Returns number of batches to acquire from the acquisition function.
        """
        n_next = self.acquisition.batches_left
        n_next = min(n_next, self.max_parallel_acquisitions - n_pending)
        indexes = list(range(current_batch_index, current_batch_index + n_next))
        return n_next, indexes

    def submit_n_batches(self, n_batches, current_batch_index=0):
        context = self.model.computation_context
        batches = list(range(current_batch_index, current_batch_index + n_batches))
        self.batches.submit_batches(batches, self.compiled_net, context)

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

