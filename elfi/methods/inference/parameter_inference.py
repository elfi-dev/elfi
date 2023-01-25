"""This module contains base class for inference methods."""

import logging
from math import ceil

import numpy as np

import elfi.client
from elfi.methods.utils import arr2d_to_batch, batch_to_arr2d
from elfi.model.elfi_model import ComputationContext, ElfiModel, NodeReference, Summary
from elfi.visualization.visualization import ProgressBar

logger = logging.getLogger(__name__)


# TODO: refactor the plotting functions

class ParameterInference:
    """A base class for parameter inference methods.

    Attributes
    ----------
    model : elfi.ElfiModel
        The ELFI graph used by the algorithm
    output_names : list
        Names of the nodes whose outputs are included in the batches
    client : elfi.client.ClientBase
        The batches are computed in the client
    max_parallel_batches : int
    state : dict
        Stores any changing data related to achieving the objective. Must include a key
        ``n_batches`` for determining when the inference is finished.
    objective : dict
        Holds the data for the algorithm to internally determine how many batches are still
        needed. You must have a key ``n_batches`` here. By default the algorithm finished when
        the ``n_batches`` in the state dictionary is equal or greater to the corresponding
        objective value.
    batches : elfi.client.BatchHandler
        Helper class for submitting batches to the client and keeping track of their
        indexes.
    pool : elfi.store.OutputPool
        Pool object for storing and reusing node outputs.


    """

    def __init__(self,
                 model,
                 output_names,
                 batch_size=1,
                 seed=None,
                 pool=None,
                 max_parallel_batches=None):
        """Construct the inference algorithm object.

        If you are implementing your own algorithm do not forget to call `super`.

        Parameters
        ----------
        model : elfi.ElfiModel
            Model to perform the inference with.
        output_names : list
            Names of the nodes whose outputs will be requested from the ELFI graph.
        batch_size : int, optional
            The number of parameter evaluations in each pass through the ELFI graph.
            When using a vectorized simulator, using a suitably large batch_size can provide
            a significant performance boost.
        seed : int, optional
            Seed for the data generation from the ElfiModel
        pool : elfi.store.OutputPool, optional
            OutputPool both stores and provides precomputed values for batches.
        max_parallel_batches : int, optional
            Maximum number of batches allowed to be in computation at the same time.
            Defaults to number of cores in the client


        """
        model = model.model if isinstance(model, NodeReference) else model
        if not model.parameter_names:
            raise ValueError('Model {} defines no parameters'.format(model))

        self.model = model.copy()
        self.output_names = self._check_outputs(output_names)

        self.client = elfi.client.get_client()

        # Prepare the computation_context
        context = ComputationContext(
            batch_size=batch_size, seed=seed, pool=pool)
        self.batches = elfi.client.BatchHandler(
            self.model, context=context, output_names=output_names, client=self.client)
        self.computation_context = context
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
        self.objective = dict()
        self.progress_bar = ProgressBar(prefix='Progress', suffix='Complete',
                                        decimals=1, length=50, fill='=')

    @property
    def pool(self):
        """Return the output pool of the inference."""
        return self.computation_context.pool

    @property
    def seed(self):
        """Return the seed of the inference."""
        return self.computation_context.seed

    @property
    def parameter_names(self):
        """Return the parameters to be inferred."""
        return self.model.parameter_names

    @property
    def batch_size(self):
        """Return the current batch_size."""
        return self.computation_context.batch_size

    def set_objective(self, *args, **kwargs):
        """Set the objective of the inference.

        This method sets the objective of the inference (values typically stored in the
        `self.objective` dict).

        Returns
        -------
        None

        """
        raise NotImplementedError

    def extract_result(self):
        """Prepare the result from the current state of the inference.

        ELFI calls this method in the end of the inference to return the result.

        Returns
        -------
        result : elfi.methods.result.Result

        """
        raise NotImplementedError

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        ELFI calls this method when a new batch has been computed and the state of
        the inference should be updated with it. It is also possible to bypass ELFI and
        call this directly to update the inference.

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
        self.state['n_batches'] += 1
        self.state['n_sim'] += self.batch_size

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

        ELFI calls this method before submitting a new batch with an increasing index
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
            Keys should match to node names in the model. These values will override any
            default values or operations in those nodes.

        """
        pass

    def plot_state(self, **kwargs):
        """Plot the current state of the algorithm.

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

    def infer(self, *args, vis=None, bar=True, **kwargs):
        """Set the objective and start the iterate loop until the inference is finished.

        See the other arguments from the `set_objective` method.

        Parameters
        ----------
        vis : dict, optional
            Plotting options. More info in self.plot_state method
        bar : bool, optional
            Flag to remove (False) or keep (True) the progress bar from/in output.

        Returns
        -------
        result : Sample

        """
        vis_opt = vis if isinstance(vis, dict) else {}

        self.set_objective(*args, **kwargs)

        while not self.finished:
            self.iterate()
            if vis:
                self.plot_state(interactive=True, **vis_opt)

            if bar:
                self.progress_bar.update_progressbar(self.state['n_batches'],
                                                     self._objective_n_batches)

        self.batches.cancel_pending()
        if vis:
            self.plot_state(close=True, **vis_opt)

        return self.extract_result()

    def iterate(self):
        """Advance the inference by one iteration.

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
        while self._allow_submit(self.batches.next_index):
            next_batch = self.prepare_new_batch(self.batches.next_index)
            logger.debug("Submitting batch %d" % self.batches.next_index)
            self.batches.submit(next_batch)

        # Handle the next ready batch in succession
        batch, batch_index = self.batches.wait_next()
        logger.debug('Received batch %d' % batch_index)
        self.update(batch, batch_index)

    @property
    def finished(self):
        """Check whether objective of n_batches have been reached."""
        return self._objective_n_batches <= self.state['n_batches']

    def _allow_submit(self, batch_index):
        return (self.max_parallel_batches > self.batches.num_pending
                and self._has_batches_to_submit and (not self.batches.has_ready()))

    @property
    def _has_batches_to_submit(self):
        return self._objective_n_batches > self.state['n_batches'] + self.batches.num_pending

    @property
    def _objective_n_batches(self):
        """Check that n_batches can be computed from the objective."""
        if 'n_batches' in self.objective:
            n_batches = self.objective['n_batches']
        elif 'n_sim' in self.objective:
            n_batches = ceil(self.objective['n_sim'] / self.batch_size)
        else:
            raise ValueError(
                'Objective must define either `n_batches` or `n_sim`.')
        return n_batches

    def _extract_result_kwargs(self):
        """Extract common arguments for the ParameterInferenceResult object."""
        return {
            'method_name': self.__class__.__name__,
            'parameter_names': self.parameter_names,
            'seed': self.seed,
            'n_sim': self.state['n_sim'],
            'n_batches': self.state['n_batches']
        }

    @staticmethod
    def _resolve_model(model, target, default_reference_class=NodeReference):
        if isinstance(model, ElfiModel) and target is None:
            raise NotImplementedError(
                "Please specify the target node of the inference method")

        if isinstance(model, NodeReference):
            target = model
            model = target.model

        if isinstance(target, str):
            target = model[target]

        if not isinstance(target, default_reference_class):
            raise ValueError('Unknown target node class')

        return model, target.name

    def _check_outputs(self, output_names):
        """Filter out duplicates and check that corresponding nodes exist.

        Preserves the order.
        """
        output_names = output_names or []
        checked_names = []
        seen = set()
        for name in output_names:
            if isinstance(name, NodeReference):
                name = name.name

            if name in seen:
                continue
            elif not isinstance(name, str):
                raise ValueError(
                    'All output names must be strings, object {} was given'.format(name))
            elif not self.model.has_node(name):
                raise ValueError(
                    'Node {} output was requested, but it is not in the model.')

            seen.add(name)
            checked_names.append(name)

        return checked_names


class ModelBased(ParameterInference):
    """Base class for model-based inference methods.

    Use this base class when the inference method needs to run multiple simulations
    with the same parameter values.

    """

    def __init__(self, model, n_sim_round, feature_names=None, batch_size=None, **kwargs):
        """Initialise model-based inference.

        Parameters
        ----------
        model: ElfiModel
            ELFI graph used by the algorithm.
        n_sim_round : int
            Number of simulations carried out with each parameter combination. Must
            be a multiple of batch_size.
        feature_names : list or str, optional
            ELFI graph nodes whose outputs are collected. Must be observable.
            Defaults to all Summary nodes.
        batch_size : int, optional
            Number of simulations carried out in each pass through the ELFI graph.
            Defaults to n_sim_round.

        """
        self.n_sim_round = n_sim_round
        batch_size = batch_size or self.n_sim_round
        if n_sim_round % batch_size != 0:
            raise ValueError('n_sim_round must be a multiple of batch_size.')

        feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
        self.feature_names = feature_names or self._get_summary_names(model)
        if len(self.feature_names) == 0:
            raise ValueError('feature_names must include at least one item.')
        for node in self.feature_names:
            if node not in model.nodes:
                raise ValueError('Node {} not found in the model'.format(node))
        output_names = model.parameter_names + self.feature_names
        super().__init__(model, output_names, batch_size=batch_size, **kwargs)

        observed = [self.model[node].observed for node in self.feature_names]
        self.observed = np.column_stack(observed)
        self.state['round'] = 0
        self.state['n_sim_round'] = 0
        self.simulated = np.zeros((self.n_sim_round, self.observed.size))

    @staticmethod
    def _get_summary_names(model):
        return [node for node in model.nodes if isinstance(model[node], Summary)
                and not node.startswith('_')]

    def _init_state(self):
        """Initialise method state.

        This can be used to reset the method state between inference calls. ELFI does not
        call this automatically.

        """
        self.state['n_batches'] = 0
        self.state['n_sim'] = 0
        self.state['round'] = 0
        self.state['n_sim_round'] = 0

    def set_objective(self, rounds):
        """Set objective for inference.

        Parameters
        ----------
        rounds : int
            Number of data collection rounds.

        """
        self.objective['round'] = rounds
        self.objective['n_batches'] = rounds * int(self.n_sim_round / self.batch_size)

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super().update(batch, batch_index)

        self._merge_batch(batch)
        if self.state['n_sim_round'] == self.n_sim_round:
            self._process_simulated()
            self.state['round'] += 1
            if self.state['round'] < self.objective['round']:
                self._init_round()

    def _init_round(self):
        """Initialise a new data collection round.

        ELFI calls this method between data collection rounds. Use this method to update
        parameter values between rounds if needed.

        """
        self.state['n_sim_round'] = 0

    def _process_simulated(self):
        """Process the simulated data.

        ELFI calls this method when a data collection round is finished. Use this method to
        update inference state based on the data stored in attribute `simulated`.

        """
        raise NotImplementedError

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

        Parameters
        ----------
        batch_index: int

        Returns
        -------
        batch: dict

        """
        params = np.atleast_2d(self.current_params)
        batch_params = np.repeat(params, self.batch_size, axis=0)
        return arr2d_to_batch(batch_params, self.parameter_names)

    @property
    def current_params(self):
        """Return parameter values explored in the current round.

        Each data collection round corresponds to fixed parameter values. The values
        can be decided in advance or between rounds.

        Returns
        -------
        np.array

        """
        raise NotImplementedError

    def infer(self, *args, **kwargs):
        """Set the objective and start the iterate loop until the inference is finished.

        Initialise a new data collection round if needed.

        Returns
        -------
        result : Sample

        """
        if self.state['round'] > 0:
            self._init_round()

        return super().infer(*args, **kwargs)

    def _merge_batch(self, batch):
        simulated = batch_to_arr2d(batch, self.feature_names)
        n_sim = self.state['n_sim_round']
        self.simulated[n_sim:n_sim + self.batch_size] = simulated
        self.state['n_sim_round'] += self.batch_size

    def _allow_submit(self, batch_index):
        batch_starts_new_round = (batch_index * self.batch_size) % self.n_sim_round == 0
        if batch_starts_new_round and self.batches.has_pending:
            return False
        else:
            return super()._allow_submit(batch_index)
