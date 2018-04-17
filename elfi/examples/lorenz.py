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
from elfi.examples.ma2 import autocov


def lorenz_ode(time_point, y, params):
    """
    Generate samples from the stochastic Lorenz model.

    Parameters
    ----------
    y : numpy.ndarray of dimension px1
        The value of timeseries where we evaluate the ODE.
    params : list
        The list of parameters needed to evaluate function. In this case it is
        list of two elements - eta and theta.

    Returns
    -------
    dY_dt : np.array
        ODE for further application.
    """

    dY_dt = np.zeros(shape=(y.shape[0]))
    eta = params[0]
    theta1 = params[1]
    theta2 = params[2]
    theta = np.array([[theta1, theta2]])
    F = params[3]
    degree = theta.shape[0]
    y_k = np.ones(shape=(y.shape[0], 1))
    for i in range(1, degree):
        y_k = np.column_stack((y_k, pow(y, i)))

    g = np.sum(np.dot(y_k, theta), axis=1)

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
        k1 = time_diff * ode(time_point=timespan[i], y=timeseries_initial,
                             params=params)
        k2 = time_diff * ode(time_point=time_mid_point,
                             y=timeseries_initial + k1 / 2, params=params)
        k3 = time_diff * ode(time_point=time_mid_point,
                             y=timeseries_initial + k2 / 2, params=params)
        k4 = time_diff * ode(time_point=timespan[i + 1],
                             y=timeseries_initial + k3, params=params)
        timeseries_initial = timeseries_initial + (
                k1 + 2 * k2 + 2 * k3 + k4) / 6
        timeseries[:, i + 1] = timeseries_initial

    return timeseries


def forecast_lorenz(theta1=None, theta2=None, F=10.,
                    phi=0.4, n_obs=50, dim=40, n_timestep=160, batch_size=1,
                    initial_state=None, random_state=None):
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
    stochastic : bool, optional
        Whether to use the stochastic or deterministic Lorenz model.

    Returns
    -------
    np.ndarray
        Timeseries initiated at timeseries_arr and satisfying ode.
    """

    if not initial_state:
        initial_state = np.linspace(-1, 6, num=dim)

    timeseries_arr = [None] * n_obs

    time_steps = np.linspace(0, 4, n_timestep)

    random_state = random_state or np.random.RandomState(batch_size)

    for k in range(n_obs):

        e = random_state.normal(0, 1, initial_state.shape[0])

        eta = np.sqrt(1 - pow(phi, 2)) * e

        timeseries = np.zeros(
            shape=(initial_state.shape[0], n_timestep),
            dtype=np.float)
        timeseries[:, 0] = initial_state

        for i in range(0, n_timestep - 1):
            params = [eta, theta1, theta2, F]
            y = runge_kutta_ode_solver(ode=lorenz_ode,
                                       timespan=np.array([time_steps[i],
                                                          time_steps[i + 1]]),
                                       timeseries_initial=timeseries[:, i],
                                       params=params)
            timeseries[:, i + 1] = y[:, -1]

            eta = phi * eta + e * np.sqrt(
                1 - pow(phi, 2))

        timeseries_arr[k] = timeseries

    return timeseries_arr


def get_model(n_obs=50, true_params=None, seed_obs=None, initial_state=None,
              dim=40, F=10.):
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
    initial_state : ndarray

    Returns
    -------
    m : elfi.ElfiModel
    """

    simulator = partial(forecast_lorenz, n_obs=n_obs,
                        initial_state=initial_state, F=F, dim=dim)

    if not true_params:
        true_params = [2.1, .1]

    m = elfi.ElfiModel()
    y_obs = simulator(*true_params, n_obs=n_obs,
                      random_state=np.random.RandomState(seed_obs))

    sim_fn = elfi.tools.vectorize(simulator)
    sumstats = []

    elfi.Prior(ss.uniform, 0.5, 3.5, model=m, name='theta1')
    elfi.Prior(ss.uniform, 0, 0.3, model=m, name='theta2')
    elfi.Simulator(sim_fn, m['theta1'], m['theta2'], observed=y_obs,
                   name='Lorenz')
    sumstats.append(
        elfi.Summary(partial(np.mean, axis=1), m['Lorenz'], name='Mean'))
    sumstats.append(
        elfi.Summary(partial(np.var, axis=1), m['Lorenz'], name='Var'))
    sumstats.append(
        elfi.Summary(partial(autocov, lag=1), m['Lorenz'], name='Autocov'))

    elfi.Distance(cost_function, *sumstats, name='d')

    return m


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
    mean = np.array([np.mean(obs) for obs in observed])
    var = np.power(np.array([np.var(obs) for obs in observed]), 2)
    return np.sum(((mean - simulated) ** 2) / var)
