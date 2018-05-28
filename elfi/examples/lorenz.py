"""Example implementation of the Lorenz forecast model.
References
----------
- Ritabrata Dutta, Jukka Corander, Samuel Kaski, and Michael U. Gutmann.
  Likelihood-free inference by ratio estimation
"""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def lorenz_ode(y, params):
    """
    Generate samples from the stochastic Lorenz model.

    Parameters
    ----------
    y : numpy.ndarray of dimension px1
        The value of time series where we evaluate the ODE.
    params : list
        The list of parameters needed to evaluate function. In this case it is
        list of two elements - eta, theta1 and theta2.

    Returns
    -------
    dY_dt : np.array
        ODE for further application.
    """

    dY_dt = np.zeros_like(y)

    eta = params[0]
    theta1 = params[1]
    theta2 = params[2]

    F = params[3]

    g = theta1 + y * theta2

    dY_dt[:, 0] = (-y[:, -2] * y[:, -1] + y[:, -1] * y[:, 1] - y[:, 0] +
                   F - g[:, 0] + eta[:, 0])

    dY_dt[:, 1] = (-y[:, -1] * y[:, 0] + y[:, 0] * y[:, 2] - y[:, 1] + F -
                   g[:, 1] + eta[:, 1])

    dY_dt[:, 2:-1] = (-y[:, :-3] * y[:, 1:-2] + y[:, 1:-2] * y[:, 3:] -
                      y[:, 2:-1] + F - g[:, 2:-1] + eta[:, 2:-1])

    dY_dt[:, -1] = (-y[:, -3] * y[:, -2] + y[:, -2] * y[:, 0] - y[:, -1] + F -
                    g[:, -1] + eta[:, -1])

    return dY_dt


def runge_kutta_ode_solver(ode, time_span, y, params):
    """
    4th order Runge-Kutta ODE solver. For more description see section 6.5 at:
    Carnahan, B., Luther, H. A., and Wilkes, J. O. (1969).
    Applied Numerical Methods. Wiley, New York.

    Parameters
    ----------
    ode : function
        Ordinary differential equation function
    time_span : numpy.ndarray
        Contains the time points where the ode needs to be
        solved. The first time point corresponds to the initial value
    y : np.ndarray of dimension px1
        Initial value of the time-series, corresponds to the first value of
        time_span
    params : list of parameters
        The parameters needed to evaluate the ode, i.e. eta and theta

    Returns
    -------
    np.ndarray
        Time series initiated at y and satisfying ode solved by this solver.
    """

    k1 = time_span * ode(y, params)

    k2 = time_span * ode(y + k1 / 2, params)

    k3 = time_span * ode(y + k2 / 2, params)

    k4 = time_span * ode(y + k3, params)

    y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y


def forecast_lorenz(theta1=None, theta2=None, F=10.,
                    phi=0.4, dim=40, n_timestep=160, batch_size=1,
                    initial_state=None, random_state=None):
    """
    The forecast Lorenz model.
    Wilks, D. S. (2005). Effects of stochastic parametrizations in the
    Lorenz ’96 system. Quarterly Journal of the Royal Meteorological Society,
    131(606), 389–407.

    Parameters
    ----------
    theta1, theta2: list or numpy.ndarray, optional
        Closure parameters. If the parameter is omitted, sampled
        from the prior.
    phi : float, optional
    initial_state: numpy.ndarray, optional
        Initial state value of the time-series. The default value is None,
        which assumes a previously computed value from a full Lorenz model as
        the Initial value.
    F : float
        Force term. The default value is 10.0.

    Returns
    -------
    np.ndarray
        Timeseries initiated at timeseries_arr and satisfying ode.
    """

    if not initial_state:
        initial_state = np.zeros(shape=(batch_size, dim))

    y_prev = y = initial_state

    theta1 = np.asarray(theta1).reshape(-1, 1)

    theta2 = np.asarray(theta2).reshape(-1, 1)

    time_span = 4 / n_timestep

    random_state = random_state or np.random

    e = random_state.normal(0, 1, y.shape)

    eta = np.sqrt(1 - pow(phi, 2)) * e

    for i in range(n_timestep):
        params = (eta, theta1, theta2, F)
        y_prev = y
        y = runge_kutta_ode_solver(ode=lorenz_ode,
                                   time_span=time_span,
                                   y=y_prev,
                                   params=params)

        eta = phi * eta + e * np.sqrt(1 - pow(phi, 2))

    y = np.stack([y, y_prev], axis=1)

    return y


def get_model(true_params=None, seed_obs=None, initial_state=None, dim=40,
              F=10.):
    """Return a complete Lorenz model in inference task.
    This is a simplified example that achieves reasonable predictions.
    For more extensive treatment and description using, see:
    Hakkarainen, J., Ilin, A., Solonen, A., Laine, M., Haario, H., Tamminen,
    J., Oja, E., and Järvinen, H. (2012). On closure parameter estimation in
    chaotic systems. Nonlinear Processes in Geophysics, 19(1), 127–143.

    Parameters
    ----------
    true_params : list, optional
        Parameters with which the observed data is generated.
    seed_obs : int, optional
        Seed for the observed data generation.
    initial_state : ndarray

    Returns
    -------
    m : elfi.ElfiModel
    """

    simulator = partial(forecast_lorenz, initial_state=initial_state,
                        F=F, dim=dim)

    if not true_params:
        true_params = [2.1, .1]

    m = elfi.ElfiModel()

    y_obs = simulator(*true_params,
                      random_state=np.random.RandomState(seed_obs))
    sumstats = []

    elfi.Prior(ss.uniform, 0.5, 3., model=m, name='theta1')
    elfi.Prior(ss.uniform, 0, 0.3, model=m, name='theta2')
    elfi.Simulator(simulator, m['theta1'], m['theta2'], observed=y_obs,
                   name='Lorenz')
    sumstats.append(
        elfi.Summary(partial(np.mean, axis=1), m['Lorenz'], name='Mean'))
    sumstats.append(
        elfi.Summary(partial(np.var, axis=1), m['Lorenz'], name='Var'))
    sumstats.append(
        elfi.Summary(autocov, m['Lorenz'], name='Autocov'))

    sumstats.append(
        elfi.Summary(cov, m['Lorenz'], name='Cov'))
    sumstats.append(
        elfi.Summary(cov, x=y_obs, side='prev', lag=1, model=m['Lorenz'],
                     name='CrosscovLeft')
    )
    sumstats.append(
        elfi.Summary(cov, x=y_obs, side='next', lag=1, model=m['Lorenz'],
                     name='CrosscovRight')
    )

    elfi.Discrepancy(cost_function, *sumstats, name='d')

    return m


def cov(x, side=None, lag=1):
    """Return the covariance of Y_{k} with its neighbour Y_{k+1}.

    Parameters
    ----------
    x : np.array of size (n, m)

    Returns
    -------
    np.array of size (n,)
        The computed covariance of two vectors in statistics.
    """

    # cross-co-variance with left neighbour Y_{k-1}
    if side == 'prev':
        return (np.mean(x[:, :-1, lag:] * x[:, 1:, :-lag], axis=1) -
                np.mean(x[:, :-1, lag:], axis=1) *
                np.mean(x[:, 1:, :-lag], axis=1))

    # cross-co-variance with right neighbour Y_{k+1}
    elif side == 'next':
        return (np.mean(x[:, 1:, lag:] * x[:, :-1, :-lag], axis=1) -
                np.mean(x[:, 1:, lag:], axis=1) *
                np.mean(x[:, :-1, :-lag], axis=1))

    # co-variance with neighbour Y_{k+1}
    return (np.mean(x[:, :-1, :] * x[:, 1:, :], axis=1) -
            np.mean(x[:, :-1, :], axis=1) * np.mean(x[:, 1:, :], axis=1))


def autocov(x, lag=1):
    """Return the autocovariance.

    Assumes a (weak) univariate stationary process with mean 0.
    Realizations are in rows.

    Parameters
    ----------
    x : np.array of size (n, m)
    lag : int, optional

    Returns
    -------
    C : np.array of size (n,)

    """
    # x = np.atleast_2d(x)

    C = np.mean(x[:, lag:, :] * x[:, :-lag, :], axis=2)

    return C


def cost_function(*simulated, observed):
    """Define cost function as in Hakkarainen et al. (2012).

    Parameters
    ----------
    observed : tuple of np.arrays
    simulated : np.arrays

    Returns
    -------
    c : ndarray
        The calculated cost function
    """
    simulated = np.column_stack(simulated)
    observed = np.column_stack(observed)

    return np.sum((simulated - observed) ** 2. / observed, axis=1)
