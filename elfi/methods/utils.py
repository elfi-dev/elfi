"""This module contains utilities for methods."""

import logging
from math import ceil
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

import elfi.model.augmenter as augmenter
from elfi.clients.native import Client
from elfi.model.elfi_model import ComputationContext

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


def compute_ess(weights: Union[None, np.ndarray] = None):
    """Compute the Effective Sample Size (ESS). Weights are assumed to be unnormalized.

    Parameters
    ----------
    weights: unnormalized weights

    """
    # normalize weights
    weights = normalize_weights(weights)

    # compute ESS
    numer = np.square(np.sum(weights))
    denom = np.sum(np.square(weights))
    return numer / denom


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
        means = np.atleast_1d(np.squeeze(means))
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
class NDimBoundingBox:
    """Class for the n-dimensional bounding box built around the optimal point."""

    def __init__(self, rotation, center, limits, eps_region):
        """Class initialiser.

        Parameters
        ----------
        rotation: (D,D) rotation matrix for the Bounding Box
        center: (D,) center of the Bounding Box
        limits: np.ndarray, shape: (D,2)
            The limits of the bounding box.

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
        self.eps_region = eps_region

        self.rotation_inv = np.linalg.inv(self.rotation)

        self.volume = self._compute_volume()

    def _compute_volume(self):
        v = np.prod(- self.limits[:, 0] + self.limits[:, 1])

        if v == 0:
            logger.warning("zero volume area")
            v = 0.05
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

        # transform to bb coordinate system
        point1 = np.dot(self.rotation_inv, point) + np.dot(self.rotation_inv, -self.center)

        # Check if point is inside bounding box
        inside = True
        for i in range(point1.shape[0]):
            if (point1[i] < self.limits[i][0]) or (point1[i] > self.limits[i][1]):
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
        """Evalute the pdf.

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


def flat_array_to_dict(names, arr):
    """Map flat array to a dictionary with parameter names.

    Parameters
    ----------
    names: List[string]
        parameter names
    arr: np.array, shape: (D,)
        flat theta array

    Returns
    -------
    Dict
       dictionary with named parameters

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
    for ii, param_name in enumerate(names):
        param_dict[param_name] = np.expand_dims(arr[ii:ii + 1], 0)
    return param_dict
