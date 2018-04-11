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
from elfi.examples.gnk import euclidean_multiss
from elfi.examples.ricker import num_zeros


def lorenz_ode(time_point, y, params):
    """
    Generate samples from the stochastic Lorenz model.

    Parameters
    ----------
    y : float or np.array
        The value of timeseries where we evaluate the ODE.
    params : list
        The list of parameters needed to evaluate function. In this case it is
        list of two elements - eta and theta.
    phi : float
    stochastic : bool, optional
        Whether to use the stochastic or deterministic Lorenz model.

    Returns
    -------
    dY_dt : np.array
        ODE for further application.

    """

    dY_dt = np.zeros(shape=(y.shape[0]))
    eta = params[0]
    theta = params[1]
    F = params[2]
    degree = theta.shape[0]
    y_k = np.ones(shape=(y.shape[0], 1))
    for i in range(1, degree):
        y_k = np.column_stack((y_k, pow(y, i)))

    g = np.sum(y_k * theta, 1)

    dY_dt[0] = -y[-2] * y[-1] + y[-1] * y[1] - y[0] + F - g[0] + eta[0]
    dY_dt[1] = -y[-1] * y[0] + y[0] * y[2] - y[1] + F - g[1] + eta[1]

    for i in range(2, y.shape[0] - 1):
        dY_dt[i] = (-y[i - 2] * y[i - 1] + y[i - 1] * y[i + 1] -
                    y[i] + F - g[i] + eta[i])
    dY_dt[-1] = -y[-3] * y[-2] + y[-2] * y[1] - y[-1] + F - g[-1] + eta[-1]

    return dY_dt


def runge_kutta_ode_solver(ode, timespan, timeseries_initial, params):
    """
    4th order Runge-Kutta ODE solver. For more description see section 6.5 at:

    Carnahan, B., Luther, H. A., and Wilkes, J. O. (1969).
    Applied Numerical Methods. Wiley, New York.

    Parameters
    ----------
    ode : function
        Ordinary differential equation function
    timespan : numpy.ndarray
        Contains the time points where the ode needs to be
        solved. The first time point corresponds to the initial value
    timeseries_initial : np.ndarray of dimension px1
        Initial value of the time-series, corresponds to the first value of
        timespan
    parameter : list of parameters
        The parameters needed to evaluate the ode, i.e. eta and theta

    Returns
    -------
    np.ndarray
        Timeseries initiated at timeseries_init and satisfying ode solved by
        this solver.
    """

    timeseries = np.zeros(
        shape=(timeseries_initial.shape[0], timespan.shape[0]))
    timeseries[:, 0] = timeseries_initial

    for i in range(0, timespan.shape[0] - 1):
        time_diff = timespan[i + 1] - timespan[i]
        time_mid_point = timespan[i] + time_diff / 2
        k1 = time_diff * ode(timespan[i], timeseries_initial, params)
        k2 = time_diff * ode(time_mid_point, timeseries_initial + k1 / 2,
                             params)
        k3 = time_diff * ode(time_mid_point, timeseries_initial + k2 / 2,
                             params)
        k4 = time_diff * ode(timespan[i + 1], timeseries_initial + k3,
                             params)
        timeseries_initial = timeseries_initial + (
                k1 + 2 * k2 + 2 * k3 + k4) / 6
        timeseries[:, i + 1] = timeseries_initial

    return timeseries


def forecast_lorenz(theta=None, initial_state=None, F=None, phi=0.4, n_obs=50,
                    n_timestep=160, batch_size=1, random_state=None):
    """
    The forecast Lorenz model.

    Wilks, D. S. (2005). Effects of stochastic parametrizations in the
    Lorenz ’96 system. Quarterly Journal of the Royal Meteorological Society,
    131(606), 389–407.

    Parameters
    ----------

    n_obs : int, optional
        Number of observations.
    n_timestep : int, optional
        Number of timesteps between [0,4], where 4 corresponds to 20 days.
        The default value is 160.
    theta: list or numpy.ndarray, optional
        Closure parameters. If the parameter is omitted, sampled
        from the prior.
    phi : float, optional

    initial_state: numpy.ndarray, optional
        Initial state value of the time-series. The default value is None,
        which assumes a previously computed value from a full Lorenz model as
        the Initial value.

    F : float
        Force term.

    stochastic : bool, optional
        Whether to use the stochastic or deterministic Lorenz model.

    Returns
    -------
    np.ndarray
        Timeseries initiated at timeseries_arr and satisfying ode.
    """

    timeseries_arr = [None] * n_obs

    time_steps = np.linspace(0, 4, n_timestep)

    random_state = random_state or np.random

    for k in range(0, n_obs):

        e = random_state.randn(n_obs)

        eta = (e * np.sqrt(1 - pow(phi, 2)))

        timeseries = np.zeros(
            shape=(initial_state.shape[0], n_timestep),
            dtype=np.float)
        timeseries[:, 0] = initial_state

        for i in range(0, n_timestep - 1):
            params = [eta, theta, F]
            y = runge_kutta_ode_solver(lorenz_ode,
                                       np.array([time_steps[i],
                                                 time_steps[i + 1]]),
                                       timeseries[:, i],
                                       params)
            timeseries[:, i + 1] = y[:, -1]

            eta = phi * eta + e * np.sqrt(
                1 - pow(phi, 2))

        timeseries_arr[k] = timeseries

    return timeseries_arr


def get_model(n_obs=50, true_params=None, seed_obs=None):
    """Return a complete Lorenz model in inference task.

    This is a simplified example that achieves reasonable predictions.
    For more extensive treatment and description using, see:

    Hakkarainen, J., Ilin, A., Solonen, A., Laine, M., Haario, H., Tamminen,
    J., Oja, E., and Järvinen, H. (2012). On closure parameter estimation in
    chaotic systems. Nonlinear Processes in Geophysics, 19(1), 127–143.

    Parameters
    ----------

    n_obs : int, optional
        Number of observations.
    true_params : list, optional
        Parameters with which the observed data is generated.
    seed_obs : int, optional
        Seed for the observed data generation.

    Returns
    -------
    m : elfi.ElfiModel

    """

    simulator = partial(forecast_lorenz, n_obs=n_obs)

    if not true_params:
        initial_state = np.array(
            [6.4558, 1.1054, -1.4502, -0.1985, 1.1905, 2.3887, 5.6689, 6.7284,
             0.9301, 4.4170, 4.0959, 2.6830, 4.7102, 2.5614, -2.9621, 2.1459,
             3.5761, 8.1188, 3.7343, 3.2147, 6.3542, 4.5297, -0.4911, 2.0779,
             5.4642, 1.7152, -1.2533, 4.6262, 8.5042, 0.7487, -1.3709, -0.0520,
             1.3196, 10.0623, -2.4885, -2.1007, 3.0754, 3.4831, 3.5744, 6.5790]
        )
        true_params = [np.array([2.1, .1]), initial_state, 10.]
    m = elfi.ElfiModel()
    y_obs = simulator(*true_params, n_obs=n_obs,
                      random_state=np.random.RandomState(seed_obs))

    sim_fn = partial(simulator, n_obs=n_obs)
    sumstats = []

    elfi.Prior(ss.expon, np.e, 2, model=m, name='t1')
    elfi.Prior(ss.truncnorm, 0, 5, model=m, name='t2')
    elfi.Prior(ss.uniform, 0, 100, model=m, name='t3')
    elfi.Simulator(sim_fn, m['t1'], m['t2'], m['t3'], observed=y_obs,
                   name='Lorenz')
    sumstats.append(
        elfi.Summary(partial(np.mean, axis=1), m['Lorenz'], name='Mean'))
    sumstats.append(
        elfi.Summary(partial(np.var, axis=1), m['Lorenz'], name='Var'))
    sumstats.append(elfi.Summary(num_zeros, m['Lorenz'], name='#0'))
    elfi.Discrepancy(euclidean_multiss, *sumstats, name='d')

    return m
