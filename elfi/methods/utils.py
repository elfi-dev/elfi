"""This module contains utilities for methods."""

import logging
from math import ceil

from typing import Callable, List

import numpy as np
import scipy.stats as ss
import scipy.optimize as optim

import elfi.model.augmenter as augmenter
from elfi.clients.native import Client
from elfi.model.elfi_model import ComputationContext
from elfi.model.elfi_model import ElfiModel

logger = logging.getLogger(__name__)


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
        loaded_net.node['_random_state'] = {'output': random_state}

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
            loaded_net.node[k] = {'output': v}

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
    """Maps flat array to a named parameter dictionary.

    Parameters
    ----------
    model: ElfiModel
    arr: (D,) flat theta array

    Returns
    -------
    param_dict
    """
    res = model.generate(batch_size=1)
    param_dict = {}
    cur_ind = 0
    for param_name in model.parameter_names:
        tensor = res[param_name]
        assert isinstance(tensor, np.ndarray)
        if tensor.ndim == 2:
            dim = tensor.shape[1]
            val = arr[cur_ind:cur_ind + dim]
            cur_ind += dim
            assert isinstance(val, np.ndarray)
            assert val.ndim == 1
            param_dict[param_name] = np.expand_dims(val, 0)

        else:
            dim = 1
            val = arr[cur_ind:cur_ind + dim]
            cur_ind += dim
            assert isinstance(val, np.ndarray)
            assert val.ndim == 1
            param_dict[param_name] = val
    return param_dict


def create_deterministic_generator(model: ElfiModel, discrepancy_name, dim, u: int):
    """
    Parameters
    __________
    u: int, seed passed to model.generate

    Returns
    -------
    func: deterministic generator
    """

    def deterministic_generator(theta: np.ndarray):
        """Creates a deterministic generator by frozing the seed to a specific value.

        Parameters
        ----------
        theta: np.ndarray (D,) flattened parameters; follows the order of the parameters

        Returns
        -------
        float: the output node sample, with frozen seed, given theta
        """

        assert theta.ndim == 1
        assert theta.shape[0] == dim

        # Map flattened array of parameters to parameter names with correct shape
        param_dict = flat_array_to_dict(model, theta)

        return float(model.generate(batch_size=1, with_values=param_dict, seed=int(u))[discrepancy_name])

    return deterministic_generator


def dummy_BB_estimation(theta_0: np.ndarray, func: Callable, lim: float, step: float, dim: int, eps: float) -> np.ndarray:
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

    BB = []
    for j in range(dim):
        BB.append([])

        # right side
        point = theta_0.copy()
        v_right = 0
        for i in range(1, nof_points+1):
            point[j] += step
            if func(point) > eps:
                v_right = (i - 1) * step
                break
            if i == nof_points:
                v_right = (i - 1) * step

        # left side
        point = theta_0.copy()
        v_left = 0
        for i in range(1, 101):
            point[j] -= step
            if func(point) > eps:
                v_left = - (i - 1) * step
                break
            if i == nof_points:
                v_left = - (i - 1) * step

        BB[j].append(theta_0[j] + v_left)
        BB[j].append(theta_0[j] + v_right)

    BB = np.array(BB)
    assert BB.ndim == 2
    assert BB.shape[0] == dim
    assert BB.shape[1] == 2
    return BB


class OptimizationProblem:

    def __init__(self, ind, nuisance, func, dim):
        self.ind = ind
        self.nuisance = nuisance
        self.function = func
        self.dim = dim

        # state
        self.state = {"attempted": False,
                      "solved": False,
                      "region": False}

        self.result = None
        self.region = None

    def solve(self, init_point):
        func = self.function

        res = optim.minimize(func,
                             init_point,
                             method="Nelder-Mead",
                             tol=.01)

        if res.success:
            self.state["attempted"] = True
            self.state["solved"] = True
            self.result = res
        else:
            self.state["solved"] = False

        return res

    def build_region(self, eps):
        """Computes Bounding Box around the theta_0.

        Parameters
        ----------
        eps

        Returns
        -------

        """
        assert self.state["solved"]

        self.region = dummy_BB_estimation(theta_0=self.result.x, func=self.function, lim=100, step=0.1,
                                          dim=self.dim, eps=eps)
        self.state["region"] = True

        return self.region


def merge_solutions(problems: List[OptimizationProblem]) -> np.ndarray:
    """Creates a single np.array, containing all obtained Bounding Boxes

    Parameters
    ----------
    problems: list with OptimizationProblem objects

    Returns
    -------
    np.ndarray (N1, dim, 2)
    """
    BB = []
    funcs = []
    for i, prob in enumerate(problems):
        if prob.state["region"]:
            BB.append(prob.region)
            funcs.append(prob.function)
    BB = np.array(BB)
    return BB, funcs


class ROMC_posterior:

    def __init__(self, optim_problems: List[Callable], prior: ModelPrior, eps: float):

        self.optim_problems = optim_problems
        self.regions, self.funcs = merge_solutions(optim_problems)
        self.prior = prior
        self.eps = eps

    def pdf_unnorm_single_point(self, theta: np.ndarray) -> float:
        """

        Parameters
        ----------
        theta: (D,)

        Returns
        -------
        unnormalized pdf evaluation
        """

        BB = self.regions
        det_generators = self.funcs
        eps = self.eps
        prior = self.prior

        # indicator if it is on bounding box, if it is inside all deterministic funcs, and volume of the box
        tmp = 0
        vol = 0
        for i in range(BB.shape[0]):
            indicator_f = 1
            # if det_generators[i](theta) > eps:
            #     indicator_f = 0

            if indicator_f == 1:
                indicator_BB = 1
                vol = 1
                for j in range(BB.shape[1]):
                    vol *= BB[i][j][1] - BB[i][j][0]
                    if (theta[j] < BB[i][j][0]) or (theta[j] > BB[i][j][1]):
                        indicator_BB = 0
                        break

            if indicator_f:
                if indicator_BB:
                    tmp += 1 / vol

            # print(theta, i, indicator_f, indicator_BB, vol, tmp)

        # prior
        pr = float(prior.pdf(np.expand_dims(theta, 0)))

        val = pr * tmp
        return val

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
        BS = theta.shape[0]
        D = theta.shape[1]

        # iterate over all points
        pdf_eval = []
        for i in range(BS):
            pdf_eval.append(self.pdf_unnorm_single_point(theta[i]))
        return np.array(pdf_eval)
