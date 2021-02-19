"""This module contains utilities for methods."""

import logging
from functools import partial
from math import ceil

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
    V_2 = np.sum(weights**2)

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


def weighted_sample_quantile(x, alpha, weights=None):
    """Calculate alpha-quantile of a weighted sample.

    Parameters
    ----------
    x : array
        One-dimensional sample
    alpha : float
        Probability threshold for alpha-quantile
    weights : array, optional
        Sample weights (possibly unnormalized), equal weights by default

    Returns
    -------
    alpha_q : array
        alpha-quantile

    """
    index = np.argsort(x)
    if weights is None:
        weights = np.ones(len(index))
    weights = weights / np.sum(weights)
    sorted_weights = weights[index]
    cum_weights = np.cumsum(sorted_weights)
    cum_weights[-1] = 1.0
    index_alpha = np.where(cum_weights >= alpha)[0][0]
    alpha_q = x[index[index_alpha]]

    return alpha_q


class DensityRatioEstimation:
    """A density ratio estimation class."""

    def __init__(self,
                 n=100,
                 epsilon=0.1,
                 max_iter=500,
                 abs_tol=0.01,
                 conv_check_interval=20,
                 fold=5,
                 optimize=False):
        """Construct the density ratio estimation algorithm object.

        Parameters
        ----------
        n : int
            Number of RBF basis functions.
        epsilon : float
            Parameter determining speed of gradient descent.
        max_iter : int
            Maximum number of iterations used in gradient descent optimization of the weights.
        abs_tol : float
            Absolute tolerance value for determining convergence of optimization of the weights.
        conv_check_interval : int
            Integer defining the interval of convergence checks in gradient descent.
        fold : int
            Number of folds in likelihood cross validation used to optimize basis scale-params.
        optimize : boolean
            Boolean indicating whether or not to optimize RBF scale.

        """
        self.n = n
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.fold = fold
        self.sigma = None
        self.conv_check_interval = conv_check_interval
        self.optimize = False

    def fit(self,
            x,
            y,
            weights_x=None,
            weights_y=None,
            sigma=None):
        """Fit the density ratio estimation object.

        Parameters
        ----------
        x : array
            Sample from the nominator distribution.
        y : sample
            Sample from the denominator distribution.
        weights_x : array
            Vector of non-negative nominator sample weights, must be able to normalize.
        weights_y : array
            Vector of non-negative denominator sample weights, must be able to normalize.
        sigma : float or list
            List of RBF kernel scales, fit selected at initial call.

        """
        self.x_len = x.shape[0]
        self.y_len = y.shape[0]
        self.x = x

        if self.x_len < self.n:
            raise ValueError("Number of RBFs ({}) can't be larger "
                             "than number of samples ({}).".format(self.n, self.x_len))

        self.theta = x[:self.n, :]
        if weights_x is None:
            weights_x = np.ones(self.x_len)
        if weights_y is None:
            weights_y = np.ones(self.y_len)
        weights_x = np.ones(self.x_len)
        weights_y = np.ones(self.y_len)
        self.weights_x = weights_x / np.sum(weights_x)
        self.weights_y = weights_y / np.sum(weights_y)

        self.x0 = np.average(x, axis=0, weights=weights_x)

        if isinstance(sigma, float):
            self.sigma = sigma
            self.optimize = False
        if self.optimize:
            if isinstance(sigma, list):
                scores_tuple = zip(*[self._KLIEP_lcv(x, y, sigma_i)
                                   for sigma_i in sigma])

                self.sigma = sigma[np.argmax(scores_tuple)]
            else:
                raise ValueError("To optimize RBF scale, "
                                 "you need to provide a list of candidate scales.")

        if self.sigma is None:
            raise ValueError("RBF width (sigma) has to provided in first call.")

        A = self._compute_A(x, self.sigma)
        b, b_normalized = self._compute_b(y, self.sigma)

        alpha = self._KLIEP(A, b, b_normalized, weights_x, self.sigma)
        self.w = partial(self._weighted_basis_sum, sigma=self.sigma, alpha=alpha)

    def _gaussian_basis(self, x, x0, sigma):
        """N-D RBF basis-function with equal scale-parameter for every dim."""
        return np.exp(-0.5 * np.sum((x - x0) ** 2) / sigma / sigma)

    def _weighted_basis_sum(self, x, sigma, alpha):
        """Weighted sum of gaussian basis functions evaluated at x."""
        return np.dot(np.array([[self._gaussian_basis(j, i, sigma) for j in self.theta]
                                for i in np.atleast_2d(x)]), alpha)

    def _compute_A(self, x, sigma):
        A = np.array([[self._gaussian_basis(i, j, sigma) for j in self.theta] for i in x])
        return A

    def _compute_b(self, y, sigma):
        b = np.sum(np.array(
                [[self._gaussian_basis(i, y[j, :], sigma) * self.weights_y[j]
                  for j in np.arange(self.y_len)]
                 for i in self.theta]), axis=1)
        b_normalized = b / np.dot(b.T, b)
        return b, b_normalized

    def _KLIEP_lcv(self, x, y, sigma):
        """Compute KLIEP scores for fold-folds."""
        A = self._compute_A(x, sigma)
        b, b_normalized = self._compute_b(y, sigma)

        fold_indices = np.array_split(np.arange(self.x_len), self.fold)
        score = np.zeros(self.fold)
        for i_fold, fold_index in enumerate(fold_indices):
            fold_index_minus = np.setdiff1d(np.arange(self.x_len), fold_index)
            alpha = self._KLIEP(A=A[fold_index_minus, :], b=b, b_normalized=b_normalized,
                                weights_x=self.weights_x[fold_index_minus], sigma=sigma)
            score[i_fold] = np.average(
                np.log(self._weighted_basis_sum(x[fold_index, :], sigma, alpha)),
                weights=self.weights_x[fold_index])

        return [np.mean(score)]

    def _KLIEP(self, A, b, b_normalized, weights_x, sigma):
        """Kullback-Leibler Importance Estimation Procedure using gradient descent."""
        alpha = 1 / self.n * np.ones(self.n)
        target_fun_prev = self._weighted_basis_sum(x=self.x, sigma=sigma, alpha=alpha)
        abs_diff = 0.0
        for i in np.arange(self.max_iter):
            dAdalpha = np.dot(A.T, (weights_x / (np.dot(A, alpha))))
            alpha += self.epsilon * dAdalpha
            alpha = np.maximum(0, alpha + (1 - np.dot(b.T, alpha)) * b_normalized)
            alpha = alpha / np.dot(b.T, alpha)
            if np.remainder(i, self.conv_check_interval) == 0:
                target_fun = self._weighted_basis_sum(x=self.x, sigma=sigma, alpha=alpha)
                abs_diff = np.linalg.norm(target_fun - target_fun_prev)
                if abs_diff < self.abs_tol:
                    break
                target_fun_prev = target_fun

        return alpha

    def max_ratio(self):
        """Find the maximum of the density ratio at numerator sample."""
        max_value = np.max(self.w(self.x))
        return max_value
