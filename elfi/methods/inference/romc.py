"""This module contains ROMC class."""

__all__ = ['ROMC']

import logging
import math
import timeit
import typing
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import scipy.optimize as optim
import scipy.spatial as spatial
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import elfi.visualization.interactive as visin
import elfi.visualization.visualization as vis
from elfi.methods.bo.acquisition import LCBSC
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.utils import stochastic_optimization
from elfi.methods.inference.parameter_inference import ParameterInference
from elfi.methods.posteriors import RomcPosterior
from elfi.methods.results import OptimizationResult, RomcSample
from elfi.methods.utils import (arr2d_to_batch, batch_to_arr2d, ceil_to_batch_size,
                                compute_ess, flat_array_to_dict)
from elfi.model.elfi_model import ElfiModel, NodeReference
from elfi.model.extensions import ModelPrior
from elfi.visualization.visualization import ProgressBar

logger = logging.getLogger(__name__)


class BoDetereministic:
    """Base class for applying Bayesian Optimisation to a deterministic objective function.

    This class (a) optimizes the determinstic function and (b) fits
    a surrogate model in the area around the optimal point. This class follows the structure
    of BayesianOptimization replacing the stochastic elfi Model with a deterministic function.
    """

    def __init__(self,
                 objective,
                 prior,
                 parameter_names,
                 n_evidence,
                 target_name=None,
                 bounds=None,
                 initial_evidence=None,
                 update_interval=10,
                 target_model=None,
                 acquisition_method=None,
                 acq_noise_var=0,
                 exploration_rate=10,
                 batch_size=1,
                 async_acq=False,
                 seed=None):
        """Initialize Bayesian optimization.

        Parameters
        ----------
        objective : Callable(np.ndarray) -> float
            The objective function
        prior : ModelPrior
            The prior distribution
        parameter_names : List[str]
            names of the parameters of interest
        n_evidence : int
            number of evidence points needed for the optimisation process to terminate
        target_name : str, optional
            the name of the output node of the deterministic function
        bounds : dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters: dict('parameter_name':(lower, upper), ... )`. If not passed,
            the range [0,1] is passed
        initial_evidence : int, dict, optional
            Number of initial evidence needed or a precomputed batch dict containing parameter
            and discrepancy values. Default value depends on the dimensionality.
        update_interval : int, optional
            How often to update the GP hyperparameters of the target_model
        target_model : GPyRegression, optional
        acquisition_method : Acquisition, optional
            Method of acquiring evidence points. Defaults to LCBSC.
        acq_noise_var : float or dict, optional
            Variance(s) of the noise added in the default LCBSC acquisition method.
            If a dictionary, values should be float specifying the variance for each dimension.
        exploration_rate : float, optional
            Exploration rate of the acquisition method
        batch_size : int, optional
            Elfi batch size. Defaults to 1.
        batches_per_acquisition : int, optional
            How many batches will be requested from the acquisition function at one go.
            Defaults to max_parallel_batches.
        async_acq : bool, optional
            Allow acquisitions to be made asynchronously, i.e. do not wait for all the
            results from the previous acquisition before making the next. This can be more
            efficient with a large amount of workers (e.g. in cluster environments) but
            forgoes the guarantee for the exactly same result with the same initial
            conditions (e.g. the seed). Default False.
        seed : int, optional
            seed for making the process reproducible
        **kwargs

        """
        self.det_func = objective
        self.prior = prior
        self.bounds = bounds
        self.batch_size = batch_size
        self.parameter_names = parameter_names
        self.seed = seed
        self.target_name = target_name
        self.target_model = target_model

        n_precomputed = 0
        n_initial, precomputed = self._resolve_initial_evidence(
            initial_evidence)
        if precomputed is not None:
            params = batch_to_arr2d(precomputed, self.parameter_names)
            n_precomputed = len(params)
            self.target_model.update(params, precomputed[target_name])

        self.batches_per_acquisition = 1
        self.acquisition_method = acquisition_method or LCBSC(self.target_model,
                                                              prior=self.prior,
                                                              noise_var=acq_noise_var,
                                                              exploration_rate=exploration_rate,
                                                              seed=self.seed)

        self.n_initial_evidence = n_initial
        self.n_precomputed_evidence = n_precomputed
        self.update_interval = update_interval
        self.async_acq = async_acq

        self.state = {'n_evidence': self.n_precomputed_evidence,
                      'last_GP_update': self.n_initial_evidence,
                      'acquisition': [], 'n_sim': 0, 'n_batches': 0}

        self.set_objective(n_evidence)

    def _resolve_initial_evidence(self, initial_evidence):
        # Some sensibility limit for starting GP regression
        precomputed = None
        n_required = max(10, 2 ** self.target_model.input_dim + 1)
        n_required = ceil_to_batch_size(n_required, self.batch_size)

        if initial_evidence is None:
            n_initial_evidence = n_required
        elif np.isscalar(initial_evidence):
            n_initial_evidence = int(initial_evidence)
        else:
            precomputed = initial_evidence
            n_initial_evidence = len(precomputed[self.target_name])

        if n_initial_evidence < 0:
            raise ValueError('Number of initial evidence must be positive or zero '
                             '(was {})'.format(initial_evidence))
        elif n_initial_evidence < n_required:
            logger.warning('We recommend having at least {} initialization points for '
                           'the initialization (now {})'.format(n_required, n_initial_evidence))

        if precomputed is None and (n_initial_evidence % self.batch_size != 0):
            logger.warning('Number of initial_evidence %d is not divisible by '
                           'batch_size %d. Rounding it up...' % (n_initial_evidence,
                                                                 self.batch_size))
            n_initial_evidence = ceil_to_batch_size(
                n_initial_evidence, self.batch_size)

        return n_initial_evidence, precomputed

    @property
    def n_evidence(self):
        """Return the number of acquired evidence points."""
        return self.state.get('n_evidence', 0)

    @property
    def acq_batch_size(self):
        """Return the total number of acquisition per iteration."""
        return self.batch_size * self.batches_per_acquisition

    def set_objective(self, n_evidence=None):
        """Set objective for inference.

        You can continue BO by giving a larger n_evidence.

        Parameters
        ----------
        n_evidence : int
            Number of total evidence for the GP fitting. This includes any initial
            evidence.

        """
        if n_evidence is None:
            n_evidence = self.objective.get('n_evidence', self.n_evidence)

        if n_evidence < self.n_evidence:
            logger.warning(
                'Requesting less evidence than there already exists')

        self.objective = {'n_evidence': n_evidence,
                          'n_sim': n_evidence - self.n_precomputed_evidence}

    def _extract_result_kwargs(self):
        """Extract common arguments for the ParameterInferenceResult object."""
        return {
            'method_name': self.__class__.__name__,
            'parameter_names': self.parameter_names,
            'seed': self.seed,
            'n_sim': self.state['n_sim'],
            'n_batches': self.state['n_batches']
        }

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        OptimizationResult

        """
        x_min, _ = stochastic_optimization(
            self.target_model.predict_mean, self.target_model.bounds, seed=self.seed)

        batch_min = arr2d_to_batch(x_min, self.parameter_names)
        outputs = arr2d_to_batch(self.target_model.X, self.parameter_names)
        outputs[self.target_name] = self.target_model.Y

        return OptimizationResult(
            x_min=batch_min, outputs=outputs, **self._extract_result_kwargs())

    def update(self, batch, batch_index):
        """Update the GP regression model of the target node with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        # super(BayesianOptimization, self).update(batch, batch_index)
        self.state['n_evidence'] += self.batch_size

        params = batch_to_arr2d(batch, self.parameter_names)
        self._report_batch(batch_index, params, batch[self.target_name])

        optimize = self._should_optimize()
        self.target_model.update(params, batch[self.target_name], optimize)
        if optimize:
            self.state['last_GP_update'] = self.target_model.n_evidence

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

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
        t = self._get_acquisition_index(batch_index)

        # Check if we still should take initial points from the prior
        if t < 0:
            return None, None

        # Take the next batch from the acquisition_batch
        acquisition = self.state['acquisition']
        if len(acquisition) == 0:
            acquisition = self.acquisition_method.acquire(
                self.acq_batch_size, t=t)

        batch = arr2d_to_batch(
            acquisition[:self.batch_size], self.parameter_names)
        self.state['acquisition'] = acquisition[self.batch_size:]

        return acquisition, batch

    def _get_acquisition_index(self, batch_index):
        acq_batch_size = self.batch_size * self.batches_per_acquisition
        initial_offset = self.n_initial_evidence - self.n_precomputed_evidence
        starting_sim_index = self.batch_size * batch_index

        t = (starting_sim_index - initial_offset) // acq_batch_size
        return t

    def fit(self):
        for ii in range(self.objective["n_sim"]):
            inp, next_batch = self.prepare_new_batch(ii)

            if inp is None:
                inp = self.prior.rvs(size=1)

                if inp.ndim == 1:
                    inp = np.expand_dims(inp, 0)
                next_batch = arr2d_to_batch(inp, self.parameter_names)

            y = np.array([self.det_func(np.squeeze(inp, 0))])

            next_batch[self.target_name] = y
            self.update(next_batch, ii)

            self.state['n_batches'] += 1
            self.state['n_sim'] += 1
        self.result = self.extract_result()

    def _should_optimize(self):
        current = self.target_model.n_evidence + self.batch_size
        next_update = self.state['last_GP_update'] + self.update_interval
        return current >= self.n_initial_evidence and current >= next_update

    def _report_batch(self, batch_index, params, distances):
        str = "Received batch {}:\n".format(batch_index)
        fill = 6 * ' '
        for i in range(self.batch_size):
            str += "{}{} at {}\n".format(fill, distances[i].item(), params[i])
        logger.debug(str)

    def plot_state(self, **options):
        """Plot the GP surface.

        This feature is still experimental and currently supports only 2D cases.
        """
        f = plt.gcf()
        if len(f.axes) < 2:
            f, _ = plt.subplots(1, 2, figsize=(
                13, 6), sharex='row', sharey='row')

        gp = self.target_model

        # Draw the GP surface
        visin.draw_contour(
            gp.predict_mean,
            gp.bounds,
            self.parameter_names,
            title='GP target surface',
            points=gp.X,
            axes=f.axes[0],
            **options)

        # Draw the latest acquisitions
        if options.get('interactive'):
            point = gp.X[-1, :]
            if len(gp.X) > 1:
                f.axes[1].scatter(*point, color='red')

        displays = [gp._gp]

        if options.get('interactive'):
            from IPython import display
            displays.insert(
                0,
                display.HTML('<span><b>Iteration {}:</b> Acquired {} at {}</span>'.format(
                    len(gp.Y), gp.Y[-1][0], point)))

        # Update
        visin._update_interactive(displays, options)

        def acq(x):
            return self.acquisition_method.evaluate(x, len(gp.X))

        # Draw the acquisition surface
        visin.draw_contour(
            acq,
            gp.bounds,
            self.parameter_names,
            title='Acquisition surface',
            points=None,
            axes=f.axes[1],
            **options)

        if options.get('close'):
            plt.close()

    def plot_discrepancy(self, axes=None, **kwargs):
        """Plot acquired parameters vs. resulting discrepancy.

        Parameters
        ----------
        axes : plt.Axes or arraylike of plt.Axes

        Return
        ------
        axes : np.array of plt.Axes

        """
        return vis.plot_discrepancy(self.target_model, self.parameter_names, axes=axes, **kwargs)

    def plot_gp(self, axes=None, resol=50, const=None, bounds=None, true_params=None, **kwargs):
        """Plot pairwise relationships as a matrix with parameters vs. discrepancy.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
        resol : int, optional
            Resolution of the plotted grid.
        const : np.array, optional
            Values for parameters in plots where held constant. Defaults to minimum evidence.
        bounds: list of tuples, optional
            List of tuples for axis boundaries.
        true_params : dict, optional
            Dictionary containing parameter names with corresponding true parameter values.

        Returns
        -------
        axes : np.array of plt.Axes

        """
        return vis.plot_gp(self.target_model, self.parameter_names, axes,
                           resol, const, bounds, true_params, **kwargs)


class ROMC(ParameterInference):
    """Robust Optimisation Monte Carlo inference method.

    Ikonomov, B., & Gutmann, M. U. (2019). Robust Optimisation Monte Carlo.
    http://arxiv.org/abs/1904.00670

    """

    def __init__(self,
                 model: typing.Union[ElfiModel, NodeReference],
                 bounds: typing.Union[typing.List, None] = None,
                 discrepancy_name: typing.Union[str, None] = None,
                 output_names: typing.Union[typing.List[str]] = None,
                 custom_optim_class=None,
                 parallelize: bool = False,
                 **kwargs):
        """Class constructor.

        Parameters
        ----------
        model: Model or NodeReference
            the elfi model or the output node of the graph
        bounds: List[(start,stop), ...]
            bounds of the n-dim bounding box area containing the mass of the posterior
        discrepancy_name: string, optional
            the name of the output node (obligatory, only if Model is passed as model)
        output_names: List[string]
            which node values to store during inference
        custom_optim_class: class
            Custom OptimizationProblem class provided by the user, to extend the algorithm
        parallelize: bool
            whether to parallelize all parts of the algorithm
        kwargs: Dict
            other named parameters

        """
        # define model
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)

        # set the output names
        output_names = [discrepancy_name] + model.parameter_names + (output_names or [])

        # setter
        self.discrepancy_name = discrepancy_name
        self.model = model
        self.model_prior = ModelPrior(model)
        self.dim = self.model_prior.dim
        self.bounds = bounds
        self.left_lim = np.array([bound[0] for bound in bounds],
                                 dtype=float) if bounds is not None else None
        self.right_lim = np.array([bound[1] for bound in bounds],
                                  dtype=float) if bounds is not None else None

        # holds the state of the inference process
        self.inference_state = {"_has_gen_nuisance": False,
                                "_has_defined_problems": False,
                                "_has_solved_problems": False,
                                "_has_fitted_surrogate_model": False,
                                "_has_filtered_solutions": False,
                                "_has_fitted_local_models": False,
                                "_has_estimated_regions": False,
                                "_has_defined_posterior": False,
                                "_has_drawn_samples": False,
                                "attempted": None,
                                "solved": None,
                                "accepted": None,
                                "computed_BB": None}

        # inputs passed during inference will be stored here
        self.inference_args = {"parallelize": parallelize}
        # user-defined OptimisationClass
        self.custom_optim_class = custom_optim_class

        # List of OptimisationProblem objects
        self.optim_problems = None

        # RomcPosterior object
        self.posterior = None
        # Samples drawn from RomcPosterior, np.ndarray (#accepted,n2,D),
        self.samples = None
        # weights of the samples, np.ndarray (#accepted,n2):
        self.weights = None
        # distances of the samples, np.ndarray (#accepted,n2):
        self.distances = None
        # RomcSample object
        self.result = None

        self.progress_bar = ProgressBar(prefix='Progress', suffix='Complete',
                                        decimals=1, length=50, fill='=')

        super(ROMC, self).__init__(model, output_names, **kwargs)

    def _define_objectives(self, n1, seed=None):
        """Define n1 deterministic optimisation problems, by freezing the seed of the generator."""
        # getters
        assert isinstance(n1, int)
        dim = self.dim
        param_names = self.parameter_names
        bounds = self.bounds
        model_prior = self.model_prior
        target_name = self.discrepancy_name

        # main part
        # It can sample at most 4x1E09 unique numbers
        # TODO fix to work with subseeds to remove the limit of 4x1E09 numbers
        up_lim = 2 ** 32 - 1
        nuisance = ss.randint(low=1, high=up_lim).rvs(
            size=n1, random_state=seed)

        # update state
        self.inference_state["_has_gen_nuisance"] = True
        self.inference_args["N1"] = n1
        self.inference_args["initial_seed"] = seed

        optim_problems = []
        for ind, nuisance in enumerate(nuisance):
            objective = self._freeze_seed(nuisance)
            # f = self._freeze_seed_f(nuisance)
            if self.custom_optim_class is None:
                optim_prob = OptimisationProblem(ind, nuisance, param_names, target_name,
                                                 objective, dim, model_prior, n1, bounds)
            else:
                optim_prob = self.custom_optim_class(ind=ind,
                                                     nuisance=nuisance,
                                                     parameter_names=param_names,
                                                     target_name=target_name,
                                                     objective=objective,
                                                     dim=dim,
                                                     prior=model_prior,
                                                     n1=n1,
                                                     bounds=bounds)

            optim_problems.append(optim_prob)

        # update state
        self.optim_problems = optim_problems
        self.inference_state["_has_defined_problems"] = True

    def _det_generator(self, theta, seed):
        model = self.model
        dim = self.dim
        output_node = self.discrepancy_name

        assert theta.ndim == 1
        assert theta.shape[0] == dim

        # Map flattened array of parameters to parameter names with correct shape
        param_dict = flat_array_to_dict(model.parameter_names, theta)
        dict_outputs = model.generate(
            batch_size=1, outputs=[output_node], with_values=param_dict, seed=int(seed))
        return float(dict_outputs[output_node]) ** 2

    def _freeze_seed(self, seed):
        """Freeze the model.generate with a specific seed.

        Parameters
        __________
        seed: int
            the seed passed to model.generate

        Returns
        -------
        Callable:
            the deterministic generator

        """
        return partial(self._det_generator, seed=seed)

    @staticmethod
    def _worker_solve_gradients(args):
        optim_prob, kwargs = args
        is_solved = optim_prob.solve_gradients(**kwargs)
        return optim_prob, is_solved

    @staticmethod
    def _worker_build_region(args):
        optim_prob, accepted, kwargs = args
        if accepted:
            is_built = optim_prob.build_region(**kwargs)
        else:
            is_built = False
        return optim_prob, is_built

    @staticmethod
    def _worker_fit_model(args):
        optim_prob, accepted, kwargs = args
        if accepted:
            optim_prob.fit_local_surrogate(**kwargs)
        return optim_prob

    def _solve_gradients(self, **kwargs):
        """Attempt to solve all defined optimization problems with a gradient-based optimiser.

        Parameters
        ----------
        kwargs: Dict
            all the keyword-arguments that will be passed to the optimiser
            None is obligatory,
            Optionals in the current implementation:
            * seed: for making the process reproducible
            * all valid arguments for scipy.optimize.minimize (e.g. method, jac)

        """
        assert self.inference_state["_has_defined_problems"]
        parallelize = self.inference_args["parallelize"]
        assert isinstance(parallelize, bool)

        # getters
        n1 = self.inference_args["N1"]
        optim_probs = self.optim_problems

        # main part
        solved = [False for _ in range(n1)]
        attempted = [False for _ in range(n1)]
        tic = timeit.default_timer()
        if parallelize is False:
            self.progress_bar.reinit_progressbar(reinit_msg="Solving gradients")
            for i in range(n1):
                self.progress_bar.update_progressbar(i + 1, n1)
                attempted[i] = True
                is_solved = optim_probs[i].solve_gradients(**kwargs)
                solved[i] = is_solved
        else:
            # parallel part
            pool = Pool()
            args = ((optim_probs[i], kwargs) for i in range(n1))
            new_list = pool.map(self._worker_solve_gradients, args)
            pool.close()
            pool.join()

            # return objects
            solved = [new_list[i][1] for i in range(n1)]
            self.optim_problems = [new_list[i][0] for i in range(n1)]

        toc = timeit.default_timer()
        logger.info("Time: %.3f sec" % (toc - tic))

        # update state
        self.inference_state["solved"] = solved
        self.inference_state["attempted"] = attempted
        self.inference_state["_has_solved_problems"] = True

    def _solve_bo(self, **kwargs):
        """Attempt to solve all defined optimization problems with Bayesian Optimisation.

        Parameters
        ----------
        kwargs: Dict
        * all the keyword-arguments that will be passed to the optimiser.
        None is obligatory.
        Optional, in the current implementation:,
        * "n_evidence": number of points for the process to terminate (default is 20)
        * "acq_noise_var": added noise at every query point (default is 0.1)

        """
        assert self.inference_state["_has_defined_problems"]

        # getters
        n1 = self.inference_args["N1"]
        optim_problems = self.optim_problems

        # main part
        attempted = []
        solved = []
        tic = timeit.default_timer()
        self.progress_bar.reinit_progressbar(reinit_msg="Bayesian Optimization")
        for i in range(n1):
            self.progress_bar.update_progressbar(i + 1, n1)
            attempted.append(True)
            is_solved = optim_problems[i].solve_bo(**kwargs)
            solved.append(is_solved)

        toc = timeit.default_timer()
        logger.info("Time: %.3f sec" % (toc - tic))

        # update state
        self.inference_state["attempted"] = attempted
        self.inference_state["solved"] = solved
        self.inference_state["_has_solved_problems"] = True
        self.inference_state["_has_fitted_surrogate_model"] = True

    def compute_eps(self, quantile):
        """Return the quantile distance, out of all optimal distance.

        Parameters
        ----------
        quantile: value in [0,1]

        Returns
        -------
        float

        """
        assert self.inference_state["_has_solved_problems"]
        assert isinstance(quantile, float)
        assert 0 <= quantile <= 1

        opt_probs = self.optim_problems
        dist = []
        for i in range(len(opt_probs)):
            if opt_probs[i].state["solved"]:
                dist.append(opt_probs[i].result.f_min)
        eps = np.quantile(dist, quantile)
        return eps

    def _filter_solutions(self, eps_filter):
        """Filter out the solutions over eps threshold.

        Parameters
        ----------
        eps_filter: float
            the threshold for filtering out solutions

        """
        # checks
        assert self.inference_state["_has_solved_problems"]

        # getters
        n1 = self.inference_args["N1"]
        solved = self.inference_state["solved"]
        optim_problems = self.optim_problems

        accepted = []
        for i in range(n1):
            if solved[i] and (optim_problems[i].result.f_min < eps_filter):
                accepted.append(True)
            else:
                accepted.append(False)

        # update status
        self.inference_args["eps_filter"] = eps_filter
        self.inference_state["accepted"] = accepted
        self.inference_state["_has_filtered_solutions"] = True

    def _build_boxes(self, **kwargs):
        """Estimate a bounding box for all accepted solutions.

        Parameters
        ----------
        kwargs: all the keyword-arguments that will be passed to the RegionConstructor.
        None is obligatory.
        Optionals,
        * eps_region, if not passed the eps for used in filtering will be used
        * use_surrogate, if not passed it will be set based on the
        optimisation method (gradients or bo)
        * step, the step size along the search direction, default 0.05
        * lim, max translation along the search direction, default 100

        """
        # getters
        optim_problems = self.optim_problems
        accepted = self.inference_state["accepted"]
        n1 = self.inference_args["N1"]
        parallelize = self.inference_args["parallelize"]
        assert isinstance(parallelize, bool)

        # main
        computed_bb = [False for _ in range(n1)]
        if parallelize is False:
            self.progress_bar.reinit_progressbar(reinit_msg="Building boxes")
            for i in range(n1):
                self.progress_bar.update_progressbar(i + 1, n1)
                if accepted[i]:
                    is_built = optim_problems[i].build_region(**kwargs)
                    computed_bb.append(is_built)
                else:
                    computed_bb.append(False)
        else:
            # parallel part
            pool = Pool()
            args = ((optim_problems[i], accepted[i], kwargs)
                    for i in range(n1))
            new_list = pool.map(self._worker_build_region, args)
            pool.close()
            pool.join()

            # return objects
            computed_bb = [new_list[i][1] for i in range(n1)]
            self.optim_problems = [new_list[i][0] for i in range(n1)]

        # update status
        self.inference_state["computed_BB"] = computed_bb
        self.inference_state["_has_estimated_regions"] = True

    def _fit_models(self, **kwargs):
        # getters
        optim_problems = self.optim_problems
        accepted = self.inference_state["accepted"]
        n1 = self.inference_args["N1"]
        parallelize = self.inference_args["parallelize"]
        assert isinstance(parallelize, bool)

        # main
        if parallelize is False:
            self.progress_bar.reinit_progressbar(reinit_msg="Fitting models")
            for i in range(n1):
                self.progress_bar.update_progressbar(i + 1, n1)
                if accepted[i]:
                    optim_problems[i].fit_local_surrogate(**kwargs)
        else:
            # parallel part
            pool = Pool()
            args = ((optim_problems[i], accepted[i], kwargs)
                    for i in range(n1))
            new_list = pool.map(self._worker_fit_model, args)
            pool.close()
            pool.join()

            # return objects
            self.optim_problems = [new_list[i] for i in range(n1)]

        # update status
        self.inference_state["_has_fitted_local_models"] = True

    def _define_posterior(self, eps_cutoff):
        """Define the posterior distribution.

        Returns
        -------
        RomcPosterior

        """
        problems = self.optim_problems
        prior = self.model_prior
        eps_filter = self.inference_args["eps_filter"]
        eps_region = self.inference_args["eps_region"]
        left_lim = self.left_lim
        right_lim = self.right_lim
        use_surrogate = self.inference_state["_has_fitted_surrogate_model"]
        use_local = self.inference_state["_has_fitted_local_models"]
        parallelize = self.inference_args["parallelize"]

        # collect all constructed regions
        regions = []
        objectives = []
        objectives_actual = []
        objectives_surrogate = None if use_surrogate is False else []
        objectives_local = None if use_local is False else []
        nuisance = []
        any_surrogate_used = (use_local or use_surrogate)
        for i, prob in enumerate(problems):
            if prob.state["region"]:
                for jj, region in enumerate(prob.regions):
                    # add nuisance variable
                    nuisance.append(prob.nuisance)

                    # add region
                    regions.append(region)

                    # add actual objective
                    objectives_actual.append(prob.objective)

                    # add local and surrogate objectives if they have been computed
                    objectives_surrogate.append(prob.surrogate) if \
                        objectives_surrogate is not None else None
                    objectives_local.append(prob.local_surrogates[jj]) if \
                        objectives_local is not None else None

                    # add the objective that will be used by the posterior
                    # the policy is (a) local (b) surrogate (c) actual
                    if use_local:
                        objectives.append(prob.local_surrogates[jj])
                    if not use_local and use_surrogate:
                        objectives.append(prob.surrogate)
                    if not use_local and not use_surrogate:
                        objectives.append(prob.objective)

        self.posterior = RomcPosterior(regions, objectives, objectives_actual,
                                       objectives_surrogate, objectives_local, nuisance,
                                       any_surrogate_used, prior, left_lim, right_lim, eps_filter,
                                       eps_region, eps_cutoff, parallelize)
        self.inference_state["_has_defined_posterior"] = True

    # Training routines
    def fit_posterior(self, n1, eps_filter, use_bo=False, quantile=None, optimizer_args=None,
                      region_args=None, fit_models=False, fit_models_args=None,
                      seed=None, eps_region=None, eps_cutoff=None):
        """Execute all training steps.

        Parameters
        ----------
        n1: integer
            nof deterministic optimisation problems
        use_bo: Boolean
            whether to use Bayesian Optimisation
        eps_filter: Union[float, str]
            threshold for filtering solution or "auto" if defined by through quantile
        quantile: Union[None, float], optional
            quantile of optimal distances to set as eps if eps="auto"
        optimizer_args: Union[None, Dict]
            keyword-arguments that will be passed to the optimiser
        region_args: Union[None, Dict]
            keyword-arguments that will be passed to the regionConstructor
        seed: Union[None, int]
            seed definition for making the training process reproducible
        eps_region:
            threshold for region construction

        """
        assert isinstance(n1, int)
        assert isinstance(use_bo, bool)
        assert eps_filter == "auto" or isinstance(eps_filter, (int, float))
        if eps_filter == "auto":
            assert isinstance(quantile, (int, float))
            quantile = float(quantile)

        # (i) define and solve problems
        self.solve_problems(n1=n1, use_bo=use_bo,
                            optimizer_args=optimizer_args, seed=seed)

        # (ii) compute eps
        if isinstance(eps_filter, (int, float)):
            eps_filter = float(eps_filter)
        elif eps_filter == "auto":
            eps_filter = self.compute_eps(quantile)

        # (iii) estimate regions
        self.estimate_regions(
            eps_filter=eps_filter, use_surrogate=use_bo, region_args=region_args,
            fit_models=fit_models, fit_models_args=fit_models_args,
            eps_region=eps_region, eps_cutoff=eps_cutoff)

        # print summary of fitting
        logger.info("NOF optimisation problems : %d " %
                    np.sum(self.inference_state["attempted"]))
        logger.info("NOF solutions obtained    : %d " %
                    np.sum(self.inference_state["solved"]))
        logger.info("NOF accepted solutions    : %d " %
                    np.sum(self.inference_state["accepted"]))

    def solve_problems(self, n1, use_bo=False, optimizer_args=None, seed=None):
        """Define and solve n1 optimisation problems.

        Parameters
        ----------
        n1: integer
            number of deterministic optimisation problems to solve
        use_bo: Boolean, default: False
            whether to use Bayesian Optimisation. If False, gradients are used.
        optimizer_args: Union[None, Dict], default None
            keyword-arguments that will be passed to the optimiser.
            The argument "seed" is automatically appended to the dict.
            In the current implementation, all arguments are optional.
        seed: Union[None, int]

        """
        assert isinstance(n1, int)
        assert isinstance(use_bo, bool)

        if optimizer_args is None:
            optimizer_args = {}

        if "seed" not in optimizer_args:
            optimizer_args["seed"] = seed

        self._define_objectives(n1=n1, seed=seed)

        if not use_bo:
            logger.info("### Solving problems using a gradient-based method ###")
            tic = timeit.default_timer()
            self._solve_gradients(**optimizer_args)
            toc = timeit.default_timer()
            logger.info("Time: %.3f sec" % (toc - tic))
        elif use_bo:
            logger.info("### Solving problems using Bayesian optimisation ###")
            tic = timeit.default_timer()
            self._solve_bo(**optimizer_args)
            toc = timeit.default_timer()
            logger.info("Time: %.3f sec" % (toc - tic))

    def estimate_regions(self, eps_filter, use_surrogate=False, region_args=None,
                         fit_models=True, fit_models_args=None,
                         eps_region=None, eps_cutoff=None):
        """Filter solutions and build the N-Dimensional bounding box around the optimal point.

        Parameters
        ----------
        eps_filter: float
            threshold for filtering the solutions
        use_surrogate: Union[None, bool]
            whether to use the surrogate model for bulding the bounding box.
            if None, it will be set based on which optimisation scheme has been used.
        region_args: Union[None, Dict]
            keyword-arguments that will be passed to the regionConstructor.
            The arguments "eps_region" and "use_surrogate" are automatically appended,
            if not defined explicitly.
        fit_models: bool
            whether to fit a helping model around the optimal point
        fit_models_args: Union[None, Dict]
            arguments passed for fitting the helping models
        eps_region: Union[None, float]
            threshold for the bounding box limits. If None, it will be equal to eps_filter.
        eps_cutoff: Union[None, float]
            threshold for the indicator function. If None, it will be equal to eps_filter.

        """
        assert self.inference_state["_has_solved_problems"], "You have firstly to " \
                                                             "solve the optimization problems."
        if region_args is None:
            region_args = {}
        if fit_models_args is None:
            fit_models_args = {}
        if eps_cutoff is None:
            eps_cutoff = eps_filter
        if eps_region is None:
            eps_region = eps_filter
        if use_surrogate is None:
            use_surrogate = True if self.inference_state["_has_fitted_surrogate_model"] else False
        if "use_surrogate" not in region_args:
            region_args["use_surrogate"] = use_surrogate
        if "eps_region" not in region_args:
            region_args["eps_region"] = eps_region

        self.inference_args["eps_region"] = eps_region
        self.inference_args["eps_cutoff"] = eps_cutoff

        self._filter_solutions(eps_filter)
        nof_solved = int(np.sum(self.inference_state["solved"]))
        nof_accepted = int(np.sum(self.inference_state["accepted"]))
        logger.info("Total solutions: %d, Accepted solutions after filtering: %d" %
                    (nof_solved, nof_accepted))
        logger.info("### Estimating regions ###\n")
        tic = timeit.default_timer()
        self._build_boxes(**region_args)
        toc = timeit.default_timer()
        logger.info("Time: %.3f sec \n" % (toc - tic))

        if fit_models:
            logger.info("### Fitting local models ###\n")
            tic = timeit.default_timer()
            self._fit_models(**fit_models_args)
            toc = timeit.default_timer()
            logger.info("Time: %.3f sec \n" % (toc - tic))

        self._define_posterior(eps_cutoff=eps_cutoff)

    # Inference Routines
    def sample(self, n2, seed=None):
        """Get samples from the posterior.

        Parameters
        ----------
        n2: int
          number of samples
        seed: int,
          seed of the sampling procedure

        """
        assert self.inference_state["_has_defined_posterior"], "You must train first"

        # set the general seed
        # np.random.seed(seed)

        # draw samples
        logger.info("### Getting Samples from the posterior ###\n")
        tic = timeit.default_timer()
        self.samples, self.weights, self.distances = self.posterior.sample(
            n2, seed=None)
        toc = timeit.default_timer()
        logger.info("Time: %.3f sec \n" % (toc - tic))
        self.inference_state["_has_drawn_samples"] = True

        # define result class
        self.result = self.extract_result()

    def eval_unnorm_posterior(self, theta):
        """Evaluate the unnormalized posterior. The operation is NOT vectorized.

        Parameters
        ----------
        theta: np.ndarray (BS, D)
            the position to evaluate

        Returns
        -------
        np.array: (BS,)

        """
        # if nothing has been done, apply all steps
        assert self.inference_state["_has_defined_posterior"], "You must train first"

        # eval posterior
        assert theta.ndim == 2
        assert theta.shape[1] == self.dim

        tic = timeit.default_timer()
        result = self.posterior.pdf_unnorm_batched(theta)
        toc = timeit.default_timer()
        logger.info("Time: %.3f sec \n" % (toc - tic))
        return result

    def eval_posterior(self, theta):
        """Evaluate the normalized posterior. The operation is NOT vectorized.

        Parameters
        ----------
        theta: np.ndarray (BS, D)

        Returns
        -------
        np.array: (BS,)

        """
        assert self.inference_state["_has_defined_posterior"], "You must train first"
        assert self.bounds is not None, "You have to set the bounds in order " \
                                        "to approximate the partition function"

        # eval posterior
        assert theta.ndim == 2
        assert theta.shape[1] == self.dim

        tic = timeit.default_timer()
        result = self.posterior.pdf(theta)
        toc = timeit.default_timer()
        logger.info("Time: %.3f sec \n" % (toc - tic))
        return result

    def compute_expectation(self, h):
        """Compute an expectation, based on h.

        Parameters
        ----------
        h: Callable

        Returns
        -------
        float or np.array, depending on the return value of the Callable h

        """
        assert self.inference_state["_has_drawn_samples"], "Draw samples first"
        return self.posterior.compute_expectation(h, self.samples, self.weights)

    # Evaluation Routines
    def compute_ess(self):
        """Compute the Effective Sample Size.

        Returns
        -------
        float
          The effective sample size.

        """
        assert self.inference_state["_has_drawn_samples"]
        return compute_ess(self.result.weights)

    def compute_divergence(self, gt_posterior, bounds=None, step=0.1, distance="Jensen-Shannon"):
        """Compute divergence between ROMC posterior and ground-truth.

        Parameters
        ----------
        gt_posterior: Callable,
            ground-truth posterior, must accepted input in a batched fashion
            (np.ndarray with shape: (BS,D))
        bounds: List[(start, stop)]
            if bounds are not passed at the ROMC constructor, they can be passed here
        step: float
        distance: str
            which distance to use. must be in ["Jensen-Shannon", "KL-Divergence"]

        Returns
        -------
        float:
          The computed divergence between the distributions

        """
        assert self.inference_state["_has_defined_posterior"]
        assert distance in ["Jensen-Shannon", "KL-Divergence"]
        if bounds is None:
            assert self.bounds is not None, "You have to define the prior's " \
                                            "limits in order to compute the divergence"

        # compute limits
        left_lim = self.left_lim
        right_lim = self.right_lim
        limits = tuple([(left_lim[i], right_lim[i])
                        for i in range(len(left_lim))])

        p = self.eval_posterior
        q = gt_posterior

        dim = len(limits)
        assert dim > 0
        assert distance in ["KL-Divergence", "Jensen-Shannon"]

        if dim == 1:
            left = limits[0][0]
            right = limits[0][1]
            nof_points = int((right - left) / step)

            x = np.linspace(left, right, nof_points)
            x = np.expand_dims(x, -1)

            p_points = np.squeeze(p(x))
            q_points = np.squeeze(q(x))

        elif dim == 2:
            left = limits[0][0]
            right = limits[0][1]
            nof_points = int((right - left) / step)
            x = np.linspace(left, right, nof_points)
            left = limits[1][0]
            right = limits[1][1]
            nof_points = int((right - left) / step)
            y = np.linspace(left, right, nof_points)

            x, y = np.meshgrid(x, y)
            inp = np.stack((x.flatten(), y.flatten()), -1)

            p_points = np.squeeze(p(inp))
            q_points = np.squeeze(q(inp))
        else:
            logger.info("Computational approximation of KL Divergence on D > 2 is intractable.")
            return None

        # compute distance
        if distance == "KL-Divergence":
            return ss.entropy(p_points, q_points)
        elif distance == "Jensen-Shannon":
            return spatial.distance.jensenshannon(p_points, q_points)

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        result : Sample

        """
        if self.samples is None:
            raise ValueError('Nothing to extract')

        method_name = "ROMC"
        parameter_names = self.model.parameter_names
        discrepancy_name = self.discrepancy_name
        weights = self.weights.flatten()
        outputs = {}
        for i, name in enumerate(self.model.parameter_names):
            outputs[name] = self.samples[:, :, i].flatten()
        outputs[discrepancy_name] = self.distances.flatten()

        return RomcSample(method_name=method_name,
                          outputs=outputs,
                          parameter_names=parameter_names,
                          discrepancy_name=discrepancy_name,
                          weights=weights)

    # Inspection Routines
    def visualize_region(self, i, force_objective=False, savefig=False):
        """Plot the acceptance area of the i-th optimisation problem.

        Parameters
        ----------
        i: int,
          index of the problem
        savefig:
          None or path

        """
        def _i_to_solved_i(ii, probs):
            k = 0
            for j in range(ii):
                if probs[j].state["region"]:
                    k += 1
            return k

        if self.samples is not None:
            samples = self.samples[_i_to_solved_i(i, self.optim_problems)]
        else:
            samples = None
        self.optim_problems[i].visualize_region(force_objective, samples, savefig)

    def distance_hist(self, savefig=False, **kwargs):
        """Plot a histogram of the distances at the optimal point.

        Parameters
        ----------
        savefig: False or str, if str it must be the path to save the figure
        kwargs: Dict with arguments to be passed to the plt.hist()

        """
        assert self.inference_state["_has_solved_problems"]

        # collect all optimal distances
        opt_probs = self.optim_problems
        dist = []
        for i in range(len(opt_probs)):
            if opt_probs[i].state["solved"]:
                d = opt_probs[i].result.f_min if opt_probs[i].result.f_min > 0 else 0
                dist.append(d)

        plt.figure()
        plt.title("Histogram of distances")
        plt.ylabel("number of problems")
        plt.xlabel("distance")
        plt.hist(dist, **kwargs)

        # if savefig=path, save to the appropriate location
        if savefig:
            plt.savefig(savefig, bbox_inches='tight')
        plt.show(block=False)


class OptimisationProblem:
    """Base class for a deterministic optimisation problem."""

    def __init__(self, ind, nuisance, parameter_names, target_name,
                 objective, dim, prior, n1, bounds):
        """Class constructor.

        Parameters
        ----------
        ind: int,
            index of the optimisation problem, must be unique
        nuisance: int,
            the seed used for defining the objective
        parameter_names: List[str]
            names of the parameters
        target_name: str
            name of the output node
        objective: Callable(np.ndarray) -> float
            the objective function
        dim: int
            the dimensionality of the problem
        prior: ModelPrior
            prior distribution of the inference
        n1: int
            number of optimisation problems defined
        bounds: List[(start, stop)]
            bounds of the optimisation problem

        """
        # index of the optimisation problem
        self.ind: int = ind
        # nuisance variable that created the objective function
        self.nuisance: int = nuisance
        # the objective function
        self.objective: typing.Callable = objective
        # dimensionality of the problem
        self.dim: int = dim
        # bounds of the prior, important for the BO case
        self.bounds: np.ndarray = bounds
        # names of the parameters, important for the BO case
        self.parameter_names: typing.List[str] = parameter_names
        # name of the distance variable, needed at the BO case
        self.target_name: str = target_name
        # The prior distribution
        self.prior: ModelPrior = prior
        # total number of optimisation problems that have been defined
        self.n1: int = n1

        # state of the optimization problem
        self.state = {"attempted": False,
                      "solved": False,
                      "has_fit_surrogate": False,
                      "has_fit_local_surrogates": False,
                      "has_built_region_with_surrogate": False,
                      "region": False}

        # Bayesian Optimisation process
        self.bo_process = None
        # surrogate model fit at Bayesian Optimisation
        self.surrogate: typing.Union[typing.Callable, None] = None
        # list with local surrogate models
        self.local_surrogates: typing.Union[typing.List[typing.Callable], None] = None
        # optimisation result
        self.result: typing.Union[RomcOptimisationResult, None] = None
        # list with acceptance regions
        self.regions: typing.Union[typing.List[NDimBoundingBox], None] = None
        # threshold for building the region
        self.eps_region: typing.Union[float, None] = None
        # initial point of the optimization process
        self.initial_point: typing.Union[np.ndarray, None] = None

    def solve_gradients(self, **kwargs):
        """Solve the optimisation problem using the scipy.optimise package.

        Parameters
        ----------
        **kwargs:
            seed (int): the seed of the optimisation process
            x0 (np.ndarray): the initial point of the optimisation process, if not
                             given explicitly, a random point will be drawn from the prior
            method (str): the name of the method to be used, should be one from
                          https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
                          Default value is "Nelder-Mead"
            jac (np.ndarray): the jacobian matrix for computing the gradients analytically

        Returns
        -------
        Boolean,
            whether the optimisation reached successfully an end point

        """
        DEFAULT_ALGORITHM = "Nelder-Mead"

        # prepare inputs
        seed = kwargs["seed"] if "seed" in kwargs else None
        if "x0" not in kwargs:
            x0 = self.prior.rvs(size=self.n1, random_state=seed)[self.ind]
        else:
            x0 = kwargs["x0"]
        method = DEFAULT_ALGORITHM if "method" not in kwargs else kwargs["method"]
        jac = kwargs["jac"] if "jac" in kwargs else None

        fun = self.objective
        self.state["attempted"] = True
        try:
            res = optim.minimize(fun=fun, x0=x0, method=method, jac=jac)

            if res.success:
                self.state["solved"] = True
                hess_appr = nd.Hessian(self.objective)(res.x)
                self.result = RomcOptimisationResult(res.x, res.fun, hess_appr)
                self.initial_point = x0
                return True
            else:
                self.state["solved"] = False
                return False
        except ValueError:
            self.state["solved"] = False
            return False

    def solve_bo(self, **kwargs):
        """Solve the optimisation problem using the BoDeterministic.

        Parameters
        ----------
        **kwargs:
            n_evidence (int): number of initial points. Default: 20
            acq_noise_var (float): added noise to point acquisition. Default: 0.1

        Returns
        -------
        Boolean, whether the optimisation reached an end point

        """
        if self.bounds is not None:
            bounds = {k: self.bounds[i]
                      for (i, k) in enumerate(self.parameter_names)}
        else:
            bounds = None

        # prepare_inputs
        n_evidence = 20 if "n_evidence" not in kwargs else kwargs["n_evidence"]
        acq_noise_var = .1 if "acq_noise_var" not in kwargs else kwargs["acq_noise_var"]

        def create_surrogate_objective(trainer):
            def surrogate_objective(theta):
                return trainer.target_model.predict_mean(np.atleast_2d(theta)).item()
            return surrogate_objective

        target_model = GPyRegression(parameter_names=self.parameter_names,
                                     bounds=bounds)

        trainer = BoDetereministic(objective=self.objective,
                                   prior=self.prior,
                                   parameter_names=self.parameter_names,
                                   n_evidence=n_evidence,
                                   target_name=self.target_name,
                                   bounds=bounds,
                                   target_model=target_model,
                                   acq_noise_var=acq_noise_var)
        trainer.fit()
        self.surrogate = create_surrogate_objective(trainer)
        self.bo_process = trainer

        param_names = self.parameter_names
        x = batch_to_arr2d(trainer.result.x_min, param_names)
        x = np.squeeze(x, 0)
        x_min = x
        hess_appr = nd.Hessian(self.objective)(x_min)
        self.result = RomcOptimisationResult(x_min, self.objective(x_min), hess_appr)

        self.state["attempted"] = True
        self.state["solved"] = True
        self.state["has_fit_surrogate"] = True
        return True

    def build_region(self, **kwargs) -> bool:
        """Compute the n-dimensional Bounding Box.

        Parameters
        ----------
        kwargs:
            **eps_region (float): threshold for building the region, this kwargument is mandatory
            **use_surrogate (bool): whether to use the surrogate objective (if it is avalable)
                                    for building the proposal region
            **eta (float) (default = 1.), step along the search direction
            **K (int) (default = 10), nof refinements
            **rep_lim (int) (default = 300), nof maximum repetitions
        Returns
        -------
        bool,
            whether the region construction was successful

        """
        assert self.state["solved"]

        # set parameters for the region construction
        if "use_surrogate" in kwargs:
            use_surrogate = kwargs["use_surrogate"]
        else:
            use_surrogate = True if self.state["has_fit_surrogate"] else False
        if use_surrogate:
            assert self.surrogate is not None, \
                "You have to first fit a surrogate model, in order to use it."
        func = self.surrogate if use_surrogate else self.objective
        self.state["has_built_region_with_surrogate"] = True if use_surrogate else False
        eta = 1. if "eta" not in kwargs else kwargs["eta"]
        K = 10 if "K" not in kwargs else kwargs["K"]
        rep_lim = 300 if "rep_lim" not in kwargs else kwargs["rep_lim"]
        assert "eps_region" in kwargs, \
            "In the current build region implementation, kwargs must contain eps_region"
        eps_region = kwargs["eps_region"]
        self.eps_region = eps_region

        # construct region
        constructor = RegionConstructor(
            self.result, func, self.dim, eps_region=eps_region,
            K=K, eta=eta, rep_lim=rep_lim)
        self.regions = constructor.build()

        # update the state
        self.state["region"] = True
        return True

    def fit_local_surrogate(self, **kwargs) -> None:
        """Fit a local quadratic model for each acceptance region.

        Parameters
        ----------
        kwargs:
            **nof_samples (int): how many samples to generate for fitting the local approximator
            **use_surrogate (bool): whether to use the surrogate model (if it is avalable) for
                                    labeling the generated samples

        Returns
        -------
        None

        """
        def local_surrogate(theta, model_scikit):
            assert theta.ndim == 1
            theta = np.expand_dims(theta, 0)
            return float(model_scikit.predict(theta))

        def create_local_surrogate(model):
            return partial(local_surrogate, model_scikit=model)

        nof_samples = 20 if "nof_samples" not in kwargs else kwargs["nof_samples"]
        if "use_surrogate" not in kwargs:
            objective = self.objective
        else:
            objective = self.surrogate if kwargs["use_surrogate"] else self.objective

        # fit the surrogate model
        local_surrogates = []
        for i in range(len(self.regions)):
            # prepare dataset
            x = self.regions[i].sample(nof_samples)
            y = np.array([objective(ii) for ii in x])

            # fit model
            model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                              ('linear', LinearRegression(fit_intercept=False))])
            model = model.fit(x, y)

            local_surrogates.append(create_local_surrogate(model))

        # update the state
        self.local_surrogates = local_surrogates
        self.state["local_surrogates"] = True

    def visualize_region(self,
                         force_objective: bool = False,
                         samples: typing.Union[np.ndarray, None] = None,
                         savefig: typing.Union[str, None] = None) -> None:
        """Plot the i-th n-dimensional bounding box region.

        Parameters
        ----------
        force_objective: if enabled, enforces the use of the objective function (not the surrogate)
        samples: the samples drawn from this region
        savefig: the path for saving the plot

        Returns
        -------
        None

        """
        if not self.state["region"]:
            print("The specific optimisation problem has not been solved! Please, choose another!")
            return
        dim = self.dim

        # if has_fit_surrogate use it, except if force_objective is on
        use_objective = (not self.state["has_built_region_with_surrogate"] or force_objective)
        func = self.objective if use_objective else self.surrogate

        # choose the first region (so far, we support only one region per objective function
        region = self.regions[0]

        if dim == 1:
            vis_region_1D(func, region, self.nuisance, self.eps_region, samples, use_objective,
                          savefig)
        else:
            vis_region_2D(func, region, self.nuisance, samples, use_objective, savefig)


class RomcOptimisationResult:
    """Base class for the optimisation result of the ROMC method."""

    def __init__(self, x_min, f_min, hess_appr, jac=None, hess=None, hess_inv=None):
        """Class constructor.

        Parameters
        ----------
        x_min: np.ndarray (D,) or float
        f_min: float
        jac: np.ndarray (D,)
        hess_inv: np.ndarray (DxD)

        """
        self.x_min = np.atleast_1d(x_min)
        self.f_min = f_min
        self.hess_appr = hess_appr
        self.jac = jac
        self.hess = hess
        self.hess_inv = hess_inv


class NDimBoundingBox:
    """Class for the n-dimensional bounding box built around the optimal point."""

    def __init__(self, rotation: np.ndarray, center: np.ndarray, limits: np.ndarray) -> None:
        """Class initialiser.

        Parameters
        ----------
        rotation: shape (D,D) rotation matrix, defines the rotation of the bounding box
        center: shape (D,) center of the bounding box
        limits: shape (D,2), limits of the bounding box around the center
        i.e. limits[:,0] (left limits) are negative translations and limits[:,1] (right limits)
        are positive translations
            The structure is
            [[d1_left_shift, d1_right_shift],
            [d2_left_shift, d2_right_shift],
            ... ,
            [dD_left_shift, dD_right_shift]]

        """
        # shape assertions
        assert rotation.ndim == 2
        assert center.ndim == 1
        assert limits.ndim == 2
        assert limits.shape[1] == 2
        assert center.shape[0] == rotation.shape[0] == rotation.shape[1]

        # assert rotation matrix is full-rank i.e. invertible
        assert np.linalg.matrix_rank(rotation) == rotation.shape[0]

        # setters
        self.dim = rotation.shape[0]
        self.rotation = rotation
        self.center = center
        self.limits = self._secure_limits(limits)

        # compute rotation matrix and volume of the region
        self.rotation_inv = np.linalg.inv(self.rotation)
        self.volume = self._compute_volume()

    def _secure_limits(self, limits: np.ndarray) -> np.ndarray:
        limits = limits.astype(float)
        eps = .001
        for i in range(limits.shape[0]):
            # assert left limits are negative translations
            assert limits[i, 0] <= 0.
            # assert right limits are positive translations
            assert limits[i, 1] >= 0.

            # if in any dimension, limits too close, move them
            if math.isclose(limits[i, 0], limits[i, 1], abs_tol=eps):
                logger.warning("The limits of the " + str(i) + "-th dimension of a bounding " +
                               "box are too narrow (<= " + str(eps) + ")")
                limits[i, 0] -= eps / 2
                limits[i, 1] += eps / 2
        return limits

    def _compute_volume(self):
        v = np.prod(- self.limits[:, 0] + self.limits[:, 1])
        assert v >= 0
        return v

    def contains(self, point):
        """Check if point is inside the bounding box.

        Parameters
        ----------
        point: (D, )

        Returns
        -------
        True/False

        """
        assert point.ndim == 1
        assert point.shape[0] == self.dim

        # transform point to the bb's coordinate system
        point = np.dot(self.rotation_inv, point) + np.dot(self.rotation_inv, -self.center)

        # Check if point is inside bounding box
        inside = True
        for i in range(point.shape[0]):
            if (point[i] < self.limits[i][0]) or (point[i] > self.limits[i][1]):
                inside = False
                break
        return inside

    def sample(self, n2, seed=None):
        """Sample n2 points from the posterior.

        Parameters
        ----------
        n2: int
        seed: seed of the sampling procedure

        Returns
        -------
        np.ndarray, shape: (n2,D)

        """
        center = self.center
        limits = self.limits
        rot = self.rotation

        loc = limits[:, 0]
        scale = limits[:, 1] - limits[:, 0]

        # draw n2 samples
        theta = []
        for i in range(loc.shape[0]):
            rv = ss.uniform(loc=loc[i], scale=scale[i])
            theta.append(rv.rvs(size=(n2, 1), random_state=seed))

        theta = np.concatenate(theta, -1)
        # translate and rotate
        theta_new = np.dot(rot, theta.T).T + center

        return theta_new

    def pdf(self, theta: np.ndarray):
        """Evalute the pdf defined by the bounding box.

        Parameters
        ----------
        theta: np.ndarray (D,)

        Returns
        -------
        float

        """
        return self.contains(theta) / self.volume

    def plot(self, samples):
        """Plot the bounding box (works only if dim=1 or dim=2).

        Parameters
        ----------
        samples: np.ndarray, shape: (N, D)

        Returns
        -------
        None

        """
        R = self.rotation
        T = self.center
        lim = self.limits

        if self.dim == 1:
            plt.figure()
            plt.title("Bounding Box region")

            # plot eigenectors
            end_point = T + R[0, 0] * lim[0][0]
            plt.plot([T[0], end_point[0]], [T[1], end_point[1]], "r-o")
            end_point = T + R[0, 0] * lim[0][1]
            plt.plot([T[0], end_point[0]], [T[1], end_point[1]], "r-o")

            plt.plot(samples, np.zeros_like(samples), "bo")
            plt.legend()
            plt.show(block=False)
        else:
            plt.figure()
            plt.title("Bounding Box region")

            # plot sampled points
            plt.plot(samples[:, 0], samples[:, 1], "bo", label="samples")

            # plot eigenectors
            x = T
            x1 = T + R[:, 0] * lim[0][0]
            plt.plot([T[0], x1[0]], [T[1], x1[1]], "y-o", label="-v1")
            x3 = T + R[:, 0] * lim[0][1]
            plt.plot([T[0], x3[0]], [T[1], x3[1]], "g-o", label="v1")

            x2 = T + R[:, 1] * lim[1][0]
            plt.plot([T[0], x2[0]], [T[1], x2[1]], "k-o", label="-v2")
            x4 = T + R[:, 1] * lim[1][1]
            plt.plot([T[0], x4[0]], [T[1], x4[1]], "c-o", label="v2")

            # plot boundaries
            def plot_side(x, x1, x2):
                tmp = x + (x1 - x) + (x2 - x)
                plt.plot([x1[0], tmp[0], x2[0]], [x1[1], tmp[1], x2[1]], "r-o")

            plot_side(x, x1, x2)
            plot_side(x, x2, x3)
            plot_side(x, x3, x4)
            plot_side(x, x4, x1)

            plt.legend()
            plt.show(block=False)


class RegionConstructor:
    """Class for constructing an n-dim bounding box region."""

    def __init__(self,
                 result: RomcOptimisationResult,
                 func: typing.Callable,
                 dim: int,
                 eps_region: float,
                 K: int = 10,
                 eta: float = 1.,
                 rep_lim: int = 300):
        """Class constructor.

        Parameters
        ----------
        result: RomcOptimisationResult, output of the optimization process
        func: Callable(np.ndarray) -> float, the objective function
        dim: int, dimensionality of the problem
        eps_region: float, threshold for building the region
        K: int (default = 10), nof refinements
        eta: float (default = 1.), step along the search direction
        rep_lim: int (default = 300), nof maximum repetitions        eta

        """
        self.res = result
        self.func = func
        self.dim = dim
        self.eps_region = eps_region
        self.K = K
        self.eta = eta
        self.rep_lim = rep_lim

    def _find_rotation_vector(self, hess_appr: np.ndarray) -> np.ndarray:
        """Return the rotation vector from the hessian approximation.

        Parameters
        ----------
        hess_appr: np.ndarray (D,D)

        Returns
        -------
        rotation matrix, np.ndarray(D, D)

        """
        dim = hess_appr.shape[0]

        # find search lines from the hessian approximation
        if np.linalg.matrix_rank(hess_appr) != dim:
            hess_appr = np.eye(dim)
        eig_val, eig_vec = np.linalg.eig(hess_appr)

        # if extreme values appear, return the I matrix
        if np.isnan(np.sum(eig_vec)) or np.isinf(np.sum(eig_vec)) or (eig_vec.dtype == complex):
            logger.info("Eye matrix return as rotation.")
            eig_vec = np.eye(dim)
        if np.linalg.matrix_rank(eig_vec) < dim:
            eig_vec = np.eye(dim)
        return eig_vec

    def build(self) -> typing.List[NDimBoundingBox]:
        """Build the bounding box.

        Returns
        -------
        List[NDimBoundingBox]

        """
        res = self.res
        func = self.func
        dim = self.dim
        eps = self.eps_region
        K = self.K
        eta = self.eta
        rep_lim = self.rep_lim

        theta_0 = np.array(res.x_min, dtype=float)
        rotation = self._find_rotation_vector(res.hess_appr)

        # compute bounding box
        bounding_box = []
        for d in range(dim):
            vd = rotation[:, d]
            # negative side
            v1 = - line_search(func, theta_0.copy(), -vd, eps, K, eta, rep_lim)
            # positive side
            v2 = line_search(func, theta_0.copy(), vd, eps, K, eta, rep_lim)
            bounding_box.append([v1, v2])
        bounding_box = np.array(bounding_box)

        # shape assertions for the bounding box
        assert bounding_box.ndim == 2
        assert bounding_box.shape[0] == dim
        assert bounding_box.shape[1] == 2

        # return list of bounding boxes with one element only
        # because in the future we will support more cases
        bb = [NDimBoundingBox(rotation, theta_0, bounding_box)]
        return bb


def comp_j(f, th_star):
    # find output dimensionality
    dim = f(th_star).shape[0]

    #
    def create_f_i(f, i):
        def f_i(th_star):
            return f(th_star)[i]

        return f_i

    jacobian = []
    for i in range(dim):
        f_i = create_f_i(f, i=i)
        jacobian.append(optim.approx_fprime(th_star, f_i, 1e-7))

    jacobian = np.array(jacobian)
    return jacobian


def line_search(f, th_star, vd, eps, K=10, eta=1., rep_lim=300):
    """Line search algorithm.

    Parameters
    ----------
    f: Callable(np.ndarray) -> float, the distance function
    th_star: np.array (D,), starting point
    vd: np.array (D,) search direction
    eps: threshold
    K: int (default = 10), nof refinements
    eta: float (default = 1.), step along the search direction
    rep_lim: int (default = 300), nof maximum repetitions

    Returns
    -------
    float, offset where f(th_star + offset*vd) > eps for the first time

    """
    th = th_star.copy()
    offset = 0
    for i in range(K):

        # find limit
        rep = 0
        while f(th) < eps and rep <= rep_lim:
            th += eta * vd
            offset += eta

            rep += 1

        th -= eta * vd
        offset -= eta

        # if repetition limit has been reached, stop
        if rep > rep_lim:
            break

        # divide eta in half
        eta = eta / 2

    # if too small region, put the maximum resolution eta as boundary
    if offset <= 0:
        offset = eta

    return offset


def vis_region_1D(func, region, nuisance, eps_region, samples, is_objective, savefig):
    plt.figure()
    if is_objective:
        plt.title("Seed = %d, f = model's objective" % nuisance)
    else:
        plt.title("Seed = %d, f = BO surrogate" % nuisance)

    # plot sampled points
    if samples is not None:
        x = samples[:, 0]
        plt.plot(x, np.zeros_like(x), "bo", label="samples")

    x = np.linspace(region.center +
                    region.limits[0, 0] -
                    0.2, region.center +
                    region.limits[0, 1] +
                    0.2, 30)
    y = [func(np.atleast_1d(theta)) for theta in x]
    plt.plot(x, y, 'r--', label="distance")
    plt.plot(region.center, 0, 'ro', label="center")
    plt.xlabel("theta")
    plt.ylabel("distance")
    plt.axvspan(region.center +
                region.limits[0, 0], region.center +
                region.limits[0, 1], label="acceptance region")
    plt.axhline(eps_region, color="g", label="eps")
    plt.legend()
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show(block=False)


def vis_region_2D(func, region, nuisance, samples, is_objective, savefig):
    plt.figure()
    if is_objective:
        plt.title("Seed = %d, f = model's objective" % nuisance)
    else:
        plt.title("Seed = %d, f = BO surrogate" % nuisance)

    max_offset = np.sqrt(
        2 * (np.max(np.abs(region.limits)) ** 2)) + 0.2
    x = np.linspace(
        region.center[0] - max_offset, region.center[0] + max_offset, 30)
    y = np.linspace(
        region.center[1] - max_offset, region.center[1] + max_offset, 30)
    X, Y = np.meshgrid(x, y)

    Z = []
    for k, ii in enumerate(x):
        Z.append([])
        for kk, jj in enumerate(y):
            Z[k].append(func(np.array([X[k, kk], Y[k, kk]])))
    Z = np.array(Z)
    plt.contourf(X, Y, Z, 100, cmap="RdGy")
    plt.plot(region.center[0], region.center[1], "ro")

    # plot sampled points
    if samples is not None:
        plt.plot(samples[:, 0], samples[:, 1], "bo", label="samples")

    # plot eigenectors
    x = region.center
    x1 = region.center + region.rotation[:, 0] * region.limits[0][0]
    plt.plot([x[0], x1[0]], [x[1], x1[1]], "y-o",
             label="-v1, f(-v1)=%.2f" % (func(x1)))
    x3 = region.center + region.rotation[:, 0] * region.limits[0][1]
    plt.plot([x[0], x3[0]], [x[1], x3[1]], "g-o",
             label="v1, f(v1)=%.2f" % (func(x3)))

    x2 = region.center + region.rotation[:, 1] * region.limits[1][0]
    plt.plot([x[0], x2[0]], [x[1], x2[1]], "k-o",
             label="-v2, f(-v2)=%.2f" % (func(x2)))
    x4 = region.center + region.rotation[:, 1] * region.limits[1][1]
    plt.plot([x[0], x4[0]], [x[1], x4[1]], "c-o",
             label="v2, f(v2)=%.2f" % (func(x3)))

    # plot boundaries
    def plot_side(x, x1, x2):
        tmp = x + (x1 - x) + (x2 - x)
        plt.plot([x1[0], tmp[0], x2[0]], [x1[1], tmp[1], x2[1]], "r-o")

    plot_side(x, x1, x2)
    plot_side(x, x2, x3)
    plot_side(x, x3, x4)
    plot_side(x, x4, x1)

    plt.xlabel("theta 1")
    plt.ylabel("theta 2")

    plt.legend()
    plt.colorbar()
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show(block=False)
