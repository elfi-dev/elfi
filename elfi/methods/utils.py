"""This module contains utilities for methods."""

import logging
from math import ceil

from typing import Callable, List, Union, Dict

import numpy as np
import scipy.stats as ss
import scipy.optimize as optim

import elfi.model.augmenter as augmenter
from elfi.clients.native import Client
from elfi.model.elfi_model import ComputationContext
from elfi.model.elfi_model import ElfiModel

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt


def arr2d_to_batch(x, names):
    """Convert a 2d array to a batch dictionary columnwise.

    Parameters
    ----------
    x : np.ndarray
        2d array of values
    names : list[str]
        List of names

    Returns
    -------
    dict
        A batch dictionary

    """
    # TODO: support vector parameter nodes
    try:
        x = x.reshape((-1, len(names)))
    except BaseException:
        raise ValueError("A dimension mismatch in converting array to batch dictionary. "
                         "This may be caused by multidimensional "
                         "prior nodes that are not yet supported.")
    batch = {p: x[:, i] for i, p in enumerate(names)}
    return batch


def batch_to_arr2d(batches, names):
    """Convert batches into a single numpy array.

    Parameters
    ----------
    batches : dict or list
       A list of batches or a single batch
    names : list
       Name of outputs to include in the array. Specifies the order.

    Returns
    -------
    np.array
        2d, where columns are batch outputs

    """
    if not batches:
        return []
    if not isinstance(batches, list):
        batches = [batches]

    rows = []
    for batch_ in batches:
        rows.append(np.column_stack([batch_[n] for n in names]))

    return np.vstack(rows)


def ceil_to_batch_size(num, batch_size):
    """Calculate how many full batches in num.

    Parameters
    ----------
    num : int
    batch_size : int

    """
    return int(batch_size * ceil(num / batch_size))


def normalize_weights(weights):
    """Normalize weights to sum to unity."""
    w = np.atleast_1d(weights)
    if np.any(w < 0):
        raise ValueError("Weights must be positive")
    wsum = np.sum(weights)
    if wsum == 0:
        raise ValueError("All weights are zero")
    return w / wsum


def weighted_var(x, weights=None):
    """Unbiased weighted variance (sample variance) for the components of x.

    The weights are assumed to be non random (reliability weights).

    Parameters
    ----------
    x : np.ndarray
        1d or 2d with observations in rows
    weights : np.ndarray or None
        1d array of weights. None defaults to standard variance.

    Returns
    -------
    s2 : np.array
        1d vector of component variances

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

    """
    if weights is None:
        weights = np.ones(len(x))

    V_1 = np.sum(weights)
    V_2 = np.sum(weights ** 2)

    xbar = np.average(x, weights=weights, axis=0)
    numerator = weights.dot((x - xbar) ** 2)
    s2 = numerator / (V_1 - (V_2 / V_1))
    return s2


class GMDistribution:
    """Gaussian mixture distribution with a shared covariance matrix."""

    @classmethod
    def pdf(cls, x, means, cov=1, weights=None):
        """Evaluate the density at points x.

        Parameters
        ----------
        x : array_like
            Scalar, 1d or 2d array of points where to evaluate, observations in rows
        means : array_like
            Means of the Gaussian mixture components. It is assumed that means[0] contains
            the mean of the first gaussian component.
        weights : array_like
            1d array of weights of the gaussian mixture components
        cov : array_like, float
            A shared covariance matrix for the mixture components

        """
        means, weights = cls._normalize_params(means, weights)

        ndim = np.asanyarray(x).ndim
        if means.ndim == 1:
            x = np.atleast_1d(x)
        if means.ndim == 2:
            x = np.atleast_2d(x)

        d = np.zeros(len(x))
        for m, w in zip(means, weights):
            d += w * ss.multivariate_normal.pdf(x, mean=m, cov=cov)

        # Cast to correct ndim
        if ndim == 0 or (ndim == 1 and means.ndim == 2):
            return d.squeeze()
        else:
            return d

    @classmethod
    def logpdf(cls, x, means, cov=1, weights=None):
        """Evaluate the log density at points x.

        Parameters
        ----------
        x : array_like
            Scalar, 1d or 2d array of points where to evaluate, observations in rows
        means : array_like
            Means of the Gaussian mixture components. It is assumed that means[0] contains
            the mean of the first gaussian component.
        weights : array_like
            1d array of weights of the gaussian mixture components
        cov : array_like, float
            A shared covariance matrix for the mixture components

        """
        return np.log(cls.pdf(x, means=means, cov=cov, weights=weights))

    @classmethod
    def rvs(cls, means, cov=1, weights=None, size=1, prior_logpdf=None, random_state=None):
        """Draw random variates from the distribution.

        Parameters
        ----------
        means : array_like
            Means of the Gaussian mixture components
        cov : array_like, optional
            A shared covariance matrix for the mixture components
        weights : array_like, optional
            1d array of weights of the gaussian mixture components
        size : int or tuple or None, optional
            Number or shape of samples to draw (a single sample has the shape of `means`).
            If None, return one sample without an enclosing array.
        prior_logpdf : callable, optional
            Can be used to check validity of random variable.
        random_state : np.random.RandomState, optional

        """
        random_state = random_state or np.random
        means, weights = cls._normalize_params(means, weights)

        if size is None:
            size = 1
            no_wrap = True
        else:
            no_wrap = False

        output = np.empty((size,) + means.shape[1:])

        n_accepted = 0
        n_left = size
        trials = 0

        while n_accepted < size:
            inds = random_state.choice(len(means), size=n_left, p=weights)
            rvs = means[inds]
            perturb = ss.multivariate_normal.rvs(mean=means[0] * 0,
                                                 cov=cov,
                                                 random_state=random_state,
                                                 size=n_left)
            x = rvs + perturb

            # check validity of x
            if prior_logpdf is not None:
                x = x[np.isfinite(prior_logpdf(x))]

            n_accepted1 = len(x)
            output[n_accepted: n_accepted + n_accepted1] = x
            n_accepted += n_accepted1
            n_left -= n_accepted1

            trials += 1
            if trials == 100:
                logger.warning("SMC: It appears to be difficult to find enough valid proposals "
                               "with prior pdf > 0. ELFI will keep trying, but you may wish "
                               "to kill the process and adjust the model priors.")

        logger.debug('Needed %i trials to find %i valid samples.', trials, size)
        if no_wrap:
            return output[0]
        else:
            return output

    @staticmethod
    def _normalize_params(means, weights):
        means = np.atleast_1d(means)
        if means.ndim > 2:
            raise ValueError('means.ndim = {} but must be at most 2.'.format(means.ndim))

        if weights is None:
            weights = np.ones(len(means))
        weights = normalize_weights(weights)
        return means, weights


def numgrad(fn, x, h=None, replace_neg_inf=True):
    """Naive numeric gradient implementation for scalar valued functions.

    Parameters
    ----------
    fn
    x : np.ndarray
        A single point in 1d vector
    h : float or list
        Stepsize or stepsizes for the dimensions
    replace_neg_inf : bool
        Replace neg inf fn values with gradient 0 (useful for logpdf gradients)

    Returns
    -------
    grad : np.ndarray
        1D gradient vector

    """
    h = 0.00001 if h is None else h
    h = np.asanyarray(h).reshape(-1)

    x = np.asanyarray(x, dtype=np.float).reshape(-1)
    dim = len(x)
    X = np.zeros((dim * 3, dim))

    for i in range(3):
        Xi = np.tile(x, (dim, 1))
        np.fill_diagonal(Xi, Xi.diagonal() + (i - 1) * h)
        X[i * dim:(i + 1) * dim, :] = Xi

    f = fn(X)
    f = f.reshape((3, dim))

    if replace_neg_inf:
        if np.any(np.isneginf(f)):
            return np.zeros(dim)

    grad = np.gradient(f, *h, axis=0)
    return grad[1, :]


# TODO: check that there are no latent variables in parameter parents.
#       pdfs and gradients wouldn't be correct in those cases as it would require
#       integrating out those latent variables. This is equivalent to that all
#       stochastic nodes are parameters.
# TODO: could use some optimization
# TODO: support the case where some priors are multidimensional
class ModelPrior:
    """Construct a joint prior distribution over all the parameter nodes in `ElfiModel`."""

    def __init__(self, model):
        """Initialize a ModelPrior.

        Parameters
        ----------
        model : ElfiModel

        """
        model = model.copy()
        self.parameter_names = model.parameter_names
        self.dim = len(self.parameter_names)
        self.client = Client()

        # Prepare nets for the pdf methods
        self._pdf_node = augmenter.add_pdf_nodes(model, log=False)[0]
        self._logpdf_node = augmenter.add_pdf_nodes(model, log=True)[0]

        self._rvs_net = self.client.compile(model.source_net, outputs=self.parameter_names)
        self._pdf_net = self.client.compile(model.source_net, outputs=self._pdf_node)
        self._logpdf_net = self.client.compile(model.source_net, outputs=self._logpdf_node)

    def rvs(self, size=None, random_state=None):
        """Sample the joint prior."""
        random_state = np.random if random_state is None else random_state

        context = ComputationContext(size or 1, seed='global')
        loaded_net = self.client.load_data(self._rvs_net, context, batch_index=0)

        # Change to the correct random_state instance
        # TODO: allow passing random_state to ComputationContext seed
        loaded_net.nodes['_random_state'].update({'output': random_state})
        del loaded_net.nodes['_random_state']['operation']

        batch = self.client.compute(loaded_net)
        rvs = np.column_stack([batch[p] for p in self.parameter_names])

        if self.dim == 1:
            rvs = rvs.reshape(size or 1)

        return rvs[0] if size is None else rvs

    def pdf(self, x):
        """Return the density of the joint prior at x."""
        return self._evaluate_pdf(x)

    def logpdf(self, x):
        """Return the log density of the joint prior at x."""
        return self._evaluate_pdf(x, log=True)

    def _evaluate_pdf(self, x, log=False):
        if log:
            net = self._logpdf_net
            node = self._logpdf_node
        else:
            net = self._pdf_net
            node = self._pdf_node

        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))
        batch = self._to_batch(x)

        # TODO: we could add a seed value that would load a "random state" instance
        #       throwing an error if it is used, for instance seed="not used".
        context = ComputationContext(len(x), seed=0)
        loaded_net = self.client.load_data(net, context, batch_index=0)

        # Override
        for k, v in batch.items():
            loaded_net.nodes[k].update({'output': v})
            del loaded_net.nodes[k]['operation']

        val = self.client.compute(loaded_net)[node]
        if ndim == 0 or (ndim == 1 and self.dim > 1):
            val = val[0]

        return val

    def gradient_pdf(self, x):
        """Return the gradient of density of the joint prior at x."""
        raise NotImplementedError

    def gradient_logpdf(self, x, stepsize=None):
        """Return the gradient of log density of the joint prior at x.

        Parameters
        ----------
        x : float or np.ndarray
        stepsize : float or list
            Stepsize or stepsizes for the dimensions

        """
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))

        grads = np.zeros_like(x)

        for i in range(len(grads)):
            xi = x[i]
            grads[i] = numgrad(self.logpdf, xi, h=stepsize)

        grads[np.isinf(grads)] = 0
        grads[np.isnan(grads)] = 0

        if ndim == 0 or (ndim == 1 and self.dim > 1):
            grads = grads[0]
        return grads

    def _to_batch(self, x):
        return {p: x[:, i] for i, p in enumerate(self.parameter_names)}


def sample_object_to_dict(data, elem, skip=''):
    """Process data from self object to data dictionary to prepare for json serialization.

    Parameters
    ----------
    data : dict, required
        Stores collected data for json
    elem : dict, required
        Default data from Sample object(s)
    skip : str, optional
        Some keys in the object should be skipped, such as `outputs` or `populations`. Latter
        is skipped in case if it is already processed previously.

    """
    for key, val in elem.__dict__.items():
        # skip `outputs` because its values are in `samples` and in `discrepancies`
        if key in ['outputs', skip]:
            continue
        if key == 'meta':
            for meta_key, meta_val in elem.__dict__[key].items():
                data[meta_key] = meta_val
            continue
        data[key] = val


def numpy_to_python_type(data):
    """Convert numpy data types to python data type for json serialization.

    Parameters
    ----------
    data : dict, required
        Stores collected data for json

    """
    for key, val in data.items():
        # in data there is keys as 'samples' which is actually a dictionary
        if isinstance(val, dict):
            for nested_key, nested_val in val.items():
                is_numpy = type(nested_val)
                data_type = str(is_numpy)
                # check whether the current value has numpy data type
                if is_numpy.__module__ == np.__name__:
                    # it is enough to check that current value's name has one of these sub-strings
                    # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
                    if 'array' in data_type:
                        data[key][nested_key] = nested_val.tolist()
                    elif 'int' in data_type:
                        data[key][nested_key] = int(nested_val)
                    elif 'float' in data_type:
                        data[key][nested_key] = float(nested_val)

        is_numpy = type(val)
        data_type = str(is_numpy)
        if is_numpy.__module__ == np.__name__:
            if 'array' in data_type:
                data[key] = val.tolist()
            elif 'int' in data_type:
                data[key] = int(val)
            elif 'float' in data_type:
                data[key] = float(val)


# ROMC utils
def flat_array_to_dict(model: ElfiModel, arr: np.ndarray) -> dict:
    """Maps flat array to a dictionart with parameter names.

    Parameters
    ----------
    model: ElfiModel
    arr: (D,) flat theta array

    Returns
    -------
    param_dict
    """
    # res = model.generate(batch_size=1)
    # param_dict = {}
    # cur_ind = 0
    # for param_name in model.parameter_names:
    #     tensor = res[param_name]
    #     assert isinstance(tensor, np.ndarray)
    #     if tensor.ndim == 2:
    #         dim = tensor.shape[1]
    #         val = arr[cur_ind:cur_ind + dim]
    #         cur_ind += dim
    #         assert isinstance(val, np.ndarray)
    #         assert val.ndim == 1
    #         param_dict[param_name] = np.expand_dims(val, 0)
    #
    #     else:
    #         dim = 1
    #         val = arr[cur_ind:cur_ind + dim]
    #         cur_ind += dim
    #         assert isinstance(val, np.ndarray)
    #         assert val.ndim == 1
    #         param_dict[param_name] = val

    # TODO: This approach covers only the case where all parameters
    # TODO: are univariate variables (i.e. independent between them)
    param_dict = {}
    for ii, param_name in enumerate(model.parameter_names):
        param_dict[param_name] = np.expand_dims(arr[ii:ii+1], 0)
    return param_dict


def create_deterministic_generator(model: ElfiModel, discrepancy_name: str, dim: int, u: float):
    """
    Parameters
    __________
    u: int, seed passed to model.generate

    Returns
    -------
    func: deterministic generator
    """

    def deterministic_generator(theta: np.ndarray) -> dict:
        """Creates a deterministic generator by frozing the seed to a specific value.

        Parameters
        ----------
        theta: np.ndarray (D,) flattened parameters; follows the order of the parameters

        Returns
        -------
        dict: the output node sample, with frozen seed, given theta
        """

        assert theta.ndim == 1
        assert theta.shape[0] == dim

        # Map flattened array of parameters to parameter names with correct shape
        param_dict = flat_array_to_dict(model, theta)
        return model.generate(batch_size=1, with_values=param_dict, seed=int(u))
    return deterministic_generator


def create_output_function(det_generator: Callable, output_node: str):
    """

    Parameters
    ----------
    det_generator: Callable that procduces the output dict of values
    output_node: output node to choose

    Returns
    -------
    Callable that produces the output of the output node
    """
    def output_function(theta: np.ndarray) -> float:
        """
        Parameters
        ----------
        theta: (D,) flattened input parameters

        Returns
        -------
        float: output
        """
        return float(det_generator(theta)[output_node]) ** 2

    return output_function


class NDimBoundingBox:
    def __init__(self, rotation: np.ndarray, center: np.ndarray, limits: np.ndarray):
        """

        Parameters
        ----------
        rotation: (D,D) rotation matrix for the Bounding Box
        center: (D,) center of the Bounding Box
        limits: (D,2)
        """
        assert rotation.ndim == 2
        assert center.ndim == 1
        assert limits.ndim == 2
        assert limits.shape[1] == 2
        assert center.shape[0] == rotation.shape[0] == rotation.shape[1]

        self.rotation = rotation
        self.center = center
        self.limits = limits
        self.dim = rotation.shape[0]

        # TODO: insert some test to check that rotation, rotation_inv are sensible
        self.rotation_inv = np.linalg.inv(self.rotation)

    def contains(self, point: np.ndarray) -> bool:
        """Checks if point is inside the bounding box.

        Parameters
        ----------
        point: (D, )

        Returns
        -------
        True/False
        """
        assert point.ndim == 1
        assert point.shape[0] == self.dim

        # transform to bb coordinate system
        point1 = np.dot(self.rotation_inv, point) + np.dot(self.rotation_inv, -self.center)

        # Check if point is inside bounding box
        inside = True
        for i in range(point1.shape[0]):
            if (point1[i] < self.limits[i][0]) or (point1[i] > self.limits[i][1]):
                inside = False
                break
        return inside


class OptimisationProblem:
    def __init__(self, ind: int, nuisance: int, func: Callable, dim: int):
        """

        Parameters
        ----------
        ind: index of the optimisation problem
        nuisance: seed of the deterministic generator
        func: deterministic generator
        dim: dimensionality of the problem
        """
        self.ind: int = ind
        self.nuisance: int = nuisance
        self.function: Callable = func
        self.dim: int = dim

        # state of the optimization problems
        self.state = {"attempted": False,
                      "solved": False,
                      "region": False}

        # store as None as values
        self.result: Union[optim.OptimizeResult, None] = None
        self.region: Union[List[NDimBoundingBox], None] = None
        self.initial_point: Union[np.ndarray, None] = None

    def solve(self, init_point: np.ndarray) -> Dict:
        """

        Parameters
        ----------
        init_point: (D,)

        Returns
        -------
        res: Dictionary holding the state of the optimisation process
        """
        func = self.function
        res = optim.minimize(func,
                             init_point,
                             method="BFGS")

        if res.success:
            self.state["attempted"] = True
            self.state["solved"] = True
            self.result = res
            self.initial_point = init_point
        else:
            self.state["solved"] = False

        return res

    def build_region(self, eps: float, mode: str = "gt_full_coverage",
                     left_lim: Union[np.ndarray, None] = None,
                     right_lim: Union[np.ndarray, None] = None,
                     step: float = 0.05) -> List[NDimBoundingBox]:
        """Computes the Bounding Box stores it at region attribute.
        If mode == "gt_full_coverage" it computes all bounding boxes.

        Parameters
        ----------
        eps: threshold
        mode: name in ["gt_full_coverage", "gt_around_theta", "romc_jacobian"]
        left_lim: needed only for gt_full_coverage
        right_lim: needed only for gt_full_coverage
        step: needed for building gt_full_coverage or gt_around_theta

        Returns
        -------
        None
        """
        assert mode in ["gt_full_coverage", "gt_around_theta", "romc_jacobian"]
        assert self.state["solved"]
        if mode == "gt_around_theta":
            self.region = gt_around_theta(theta_0=self.result.x,
                                          func=self.function,
                                          lim=100,
                                          step=0.05,
                                          dim=self.dim, eps=eps)
        elif mode == "gt_full_coverage":
            assert left_lim is not None
            assert right_lim is not None
            assert self.dim <= 1

            self.region = gt_full_coverage(theta_0=self.result.x,
                                           func=self.function,
                                           left_lim=left_lim,
                                           right_lim=right_lim,
                                           step=step,
                                           eps=eps)

        elif mode == "romc_jacobian":
            self.region = romc_jacobian(theta_0=self.result.x,
                                        func=self.function,
                                        dim=self.dim,
                                        eps=eps,
                                        lim=100,
                                        step=step)

        self.state["region"] = True

        return self.region


class RomcPosterior:

    def __init__(self,
                 regions: List[NDimBoundingBox],
                 funcs: List[Callable],
                 prior: ModelPrior,
                 left_lim,
                 right_lim,
                 eps: float):

        # self.optim_problems = optim_problems
        self.regions = regions
        self.funcs = funcs
        self.prior = prior
        self.eps = eps
        self.left_lim = left_lim
        self.right_lim = right_lim
        self.dim = prior.dim
        self.partition = None

    def pdf_unnorm_single_point(self, theta: np.ndarray) -> float:
        """

        Parameters
        ----------
        theta: (D,)

        Returns
        -------
        unnormalized pdf evaluation
        """
        assert isinstance(theta, np.ndarray)
        assert theta.ndim == 1

        prior = self.prior

        tmp = self.is_inside_box(theta)
        # TODO add indicator: at 1D its ok

        # prior
        pr = float(prior.pdf(np.expand_dims(theta, 0)))

        val = pr * tmp
        return val

    def is_inside_box(self, theta: np.ndarray) -> int:
        regions = self.regions
        nof_inside = 0
        for reg in regions:
            if reg.contains(theta):
                nof_inside += 1
        return nof_inside

    def pdf_unnorm(self, theta: np.ndarray):
        """Computes the value of the unnormalized posterior. The operation is NOT vectorized.

        Parameters
        ----------
        theta: np.ndarray (BS, D)

        Returns
        -------
        np.array: (BS,)
        """
        assert isinstance(theta, np.ndarray)
        assert theta.ndim == 2
        batch_size = theta.shape[0]

        # iterate over all points
        pdf_eval = []
        for i in range(batch_size):
            pdf_eval.append(self.pdf_unnorm_single_point(theta[i]))
        return np.array(pdf_eval)

    def approximate_partition(self, nof_points: int = 200):
        """Approximates Z, computing the integral as a sum.

        Parameters
        ----------
        nof_points: int, nof points to use in each dimension
        """
        dim = self.dim
        left_lim = self.left_lim
        right_lim = self.right_lim

        partition = 0
        vol_per_point = np.prod((right_lim - left_lim) / nof_points)

        if dim == 1:
            for i in np.linspace(left_lim[0], right_lim[0], nof_points):
                theta = np.array([[i]])
                partition += self.pdf_unnorm(theta)[0] * vol_per_point
        elif dim == 2:
            for i in np.linspace(left_lim[0], right_lim[0], nof_points):
                for j in np.linspace(left_lim[1], right_lim[1], nof_points):
                    theta = np.array([[i, j]])
                    partition += self.pdf_unnorm(theta)[0] * vol_per_point
        else:
            print("ERROR: Approximate partition is not implemented for D > 2")

        # update inference state
        self.partition = partition
        return partition

    def pdf(self, theta):
        assert theta.ndim == 2
        assert theta.shape[1] == self.dim
        assert self.dim <= 2, "PDF can be computed up to 2 dimensional problems."

        if self.partition is not None:
            partition = self.partition
        else:
            partition = self.approximate_partition()
            self.partition = partition

        pdf_eval = []
        for i in range(theta.shape[0]):
            pdf_eval.append(self.pdf_unnorm(theta[i:i + 1]) / partition)
        return np.array(pdf_eval)


def collect_solutions(problems: List[OptimisationProblem]) -> (List[NDimBoundingBox], List[Callable]):
    """Prepares Bounding Boxes objects and optim functions for defining the ROMC_posterior.

    Parameters
    ----------
    problems: list with OptimizationProblem objects

    Returns
    -------
    bounding_boxes: list with Bounding Boxes objects
    funcs: list with deterministic functions
    """

    bounding_boxes = []
    funcs = []
    for i, prob in enumerate(problems):
        if prob.state["region"]:
            for jj in range(len(prob.region)):
                bounding_boxes.append(prob.region[jj])
            funcs.append(prob.function)
    return bounding_boxes, funcs


def gt_around_theta(theta_0: np.ndarray, func: Callable, lim: float, step: float, dim: int,
                    eps: float) -> List[NDimBoundingBox]:
    """Computes the Bounding Box (BB) around theta_0, such that func(x) < eps for x inside the area.
    The BB computation is done with an iterative evaluation of the func along each dimension.

    Parameters
    ----------
    theta_0: np.array (D,)
    func: callable(theta_0) -> float, the deterministic function
    lim: the maximum translation along each direction
    step: the step along each direction
    dim: the dimensionality of theta_0
    eps: float, the threshold of the distance

    Returns
    -------

    """
    assert theta_0.ndim == 1
    assert theta_0.shape[0] == dim

    # assert that best point is in limit
    assert func(theta_0) < eps

    # compute nof_points
    nof_points = int(lim / step)

    bounding_box = []
    for j in range(dim):
        bounding_box.append([])

        # right side
        point = theta_0.copy()
        v_right = 0
        for i in range(1, nof_points + 1):
            point[j] += step
            if func(point) > eps:
                v_right = (i - 1) * step
                break
            if i == nof_points:
                v_right = (i - 1) * step

        # left side
        point = theta_0.copy()
        v_left = 0
        for i in range(1, nof_points + 1):
            point[j] -= step
            if func(point) > eps:
                v_left = - (i - 1) * step
                break
            if i == nof_points:
                v_left = - (i - 1) * step

        bounding_box[j].append(theta_0[j] + v_left)
        bounding_box[j].append(theta_0[j] + v_right)

    bounding_box = np.array(bounding_box)
    assert bounding_box.ndim == 2
    assert bounding_box.shape[0] == dim
    assert bounding_box.shape[1] == 2

    # cast to bb object
    center = []
    limits = []
    for jj in range(dim):
        tmp_center = (bounding_box[jj, 0] + bounding_box[jj, 1]) / 2
        right = bounding_box[jj, 1] - tmp_center
        left = - (tmp_center - bounding_box[jj, 0])

        limits.append(np.array([left, right]))
        center.append(tmp_center)
    center = np.array(center)
    limits = np.array(limits)

    bb = [NDimBoundingBox(np.eye(dim), center, limits)]
    return bb


def gt_full_coverage(theta_0: np.ndarray,
                     func: Callable,
                     left_lim: np.ndarray,
                     right_lim: np.ndarray,
                     step: float,
                     eps: float) -> List[NDimBoundingBox]:
    """Implemented only for the 1D case, to serve as ground truth Bounding Box. It scans all values
    between [left_lim, right_lim] in order to find all sets of values inside eps.

    Parameters
    ----------
    theta_0: (1,)
    func: the deteriminstic generator
    left_lim: (1,)
    right_lim: (1,)
    step: step for moving along the axis
    eps: threshold

    Returns
    -------
    List of Bounding Box objects
    """
    # checks
    assert theta_0.ndim == 1
    assert theta_0.shape[0] == 1
    assert left_lim.ndim == 1
    assert left_lim.shape[0] == 1
    assert right_lim.ndim == 1
    assert right_lim.shape[0] == 1

    nof_points = int((right_lim[0] - left_lim[0]) / step)
    x = np.linspace(left_lim[0], right_lim[0], nof_points)
    regions = []
    opened = False
    for i, point in enumerate(x):
        if func(np.array([point])) < eps:
            if not opened:
                opened = True
                regions.append([point])
        else:
            if opened:
                opened = False
                regions[-1].append(point)
    if opened:
        regions[-1].append(point)

    # if no region is created, just add a small one around theta
    if len(regions) == 0:
        assert func(theta_0) < eps
        regions = [[theta_0[0] - step, theta_0[0] + step]]
    regions = np.expand_dims(np.concatenate(regions), 0)

    # make each region a ndimBoundingBox object
    nof_areas = int(regions.shape[1] / 2)
    areas = []
    for k in range(nof_areas):
        center = (regions[0, 2 * k + 1] + regions[0, 2 * k]) / 2
        right = regions[0, 2 * k + 1] - center
        left = - (center - regions[0, 2 * k])
        limits = np.expand_dims(np.array([left, right]), 0)
        areas.append(NDimBoundingBox(np.eye(1), np.array([center]), limits))

    return areas


def romc_jacobian(theta_0: np.ndarray, func: Callable, dim: int, eps: float,
                  lim: float, step: float):
    # TODO check in high dimensions
    h = 1e-5
    grad_vec = optim.approx_fprime(theta_0, func, h)
    grad_vec = np.expand_dims(grad_vec, -1)

    hess_appr = np.dot(grad_vec, grad_vec.T)

    assert hess_appr.shape[0] == dim
    assert hess_appr.shape[1] == dim

    eig_val, eig_vec = np.linalg.eig(hess_appr)
    rotation = eig_vec

    # compute limits
    nof_points = int(lim / step)

    bounding_box = []
    for j in range(dim):
        bounding_box.append([])
        vect = eig_vec[:, j]

        # right side
        point = theta_0.copy()
        v_right = 0
        for i in range(1, nof_points + 1):
            point[j] += step * vect
            if func(point) > eps:
                v_right = (i - 1) * step
                break
            if i == nof_points:
                v_right = (i - 1) * step

        # left side
        point = theta_0.copy()
        v_left = 0
        for i in range(1, nof_points + 1):
            point[j] -= step * vect
            if func(point) > eps:
                v_left = - (i - 1) * step
                break
            if i == nof_points:
                v_left = - (i - 1) * step

        bounding_box[j].append(v_left)
        bounding_box[j].append(v_right)

    bounding_box = np.array(bounding_box)
    assert bounding_box.ndim == 2
    assert bounding_box.shape[0] == dim
    assert bounding_box.shape[1] == 2

    bb = [NDimBoundingBox(rotation, theta_0, bounding_box)]
    return bb
