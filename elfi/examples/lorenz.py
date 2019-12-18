"""Example implementation of the Lorenz forecast model.

References
----------
- Dutta, R., Corander, J., Kaski, S. and Gutmann, M.U., 2016.
  Likelihood-free inference by ratio estimation. arXiv preprint arXiv:1611.10242.
  https://arxiv.org/abs/1611.10242

"""

from functools import partial

import numpy as np

import elfi


def _lorenz_ode(y, params):
    """Parametrized Lorenz 96 system defined by a coupled stochastic differential equations (SDE).

    Parameters
    ----------
    y : numpy.ndarray of dimension (batch_size, n_obs)
        Current state of the SDE.
    params : list
        The list of parameters needed to evaluate function. In this case it is
        list of four elements - eta, theta1, theta2 and f.

    Returns
    -------
    dy_dt : np.array
        Rate of change of the SDE.

    """
    dy_dt = np.empty_like(y)

    eta = params[0]
    theta1 = params[1]
    theta2 = params[2]

    f = params[3]

    g = theta1 + y * theta2

    dy_dt[:, 0] = -y[:, -2] * y[:, -1] + y[:, -1] * y[:, 1] - y[:, 0] + f - g[:, 0] + eta[:, 0]

    dy_dt[:, 1] = -y[:, -1] * y[:, 0] + y[:, 0] * y[:, 2] - y[:, 1] + f - g[:, 1] + eta[:, 1]

    dy_dt[:, 2:-1] = (-y[:, :-3] * y[:, 1:-2] + y[:, 1:-2] * y[:, 3:] - y[:, 2:-1] + f - g[:, 2:-1]
                      + eta[:, 2:-1])

    dy_dt[:, -1] = (-y[:, -3] * y[:, -2] + y[:, -2] * y[:, 0] - y[:, -1] + f - g[:, -1]
                    + eta[:, -1])

    return dy_dt


def runge_kutta_ode_solver(ode, time_step, y, params):
    """4th order Runge-Kutta ODE solver.

    Carnahan, B., Luther, H. A., and Wilkes, J. O. (1969).
    Applied Numerical Methods. Wiley, New York.

    Parameters
    ----------
    ode : function
        Ordinary differential equation function. In the Lorenz model it is SDE.
    time_step : float
    y : np.ndarray of dimension (batch_size, n_obs)
        Current state of the time-series.
    params : list of parameters
        The parameters needed to evaluate the ode. In this case it is
        list of four elements - eta, theta1, theta2 and f.

    Returns
    -------
    np.ndarray
        Resulting state initiated at y and satisfying ode solved by this solver.

    """
    k1 = time_step * ode(y, params)

    k2 = time_step * ode(y + k1 / 2, params)

    k3 = time_step * ode(y + k2 / 2, params)

    k4 = time_step * ode(y + k3, params)

    y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y


def forecast_lorenz(theta1=None, theta2=None, f=10., phi=0.984, n_obs=40, n_timestep=160,
                    batch_size=1, initial_state=None, random_state=None, total_duration=4):
    """Forecast Lorenz model.

    Wilks, D. S. (2005). Effects of stochastic parametrizations in the
    Lorenz ’96 system. Quarterly Journal of the Royal Meteorological Society,
    131(606), 389–407.

    Parameters
    ----------
    theta1, theta2: list or numpy.ndarray
        Closure parameters.
    phi : float, optional
        This value is used to express stochastic forcing term. It should be configured according
        to force term and eventually impacts to the result of eta.
        More details in Wilks (2005) et al.
    initial_state: numpy.ndarray, optional
        Initial state value of the time-series.
    f : float, optional
        Force term
    n_obs : int, optional
        Size of the observed 1D grid
    n_timestep : int, optional
        Number of the time step intervals
    batch_size : int, optional
    random_state : np.random.RandomState, optional
    total_duration : float, optional

    Returns
    -------
    np.ndarray of size (b, n, m) which is (batch_size, time, n_obs)
        The computed SDE with time series.

    """
    if not initial_state:
        initial_state = np.tile([2.40711741e-01, 4.75597337e+00, 1.19145654e+01, 1.31324866e+00,
                                 2.82675744e+00, 3.96016971e+00, 2.10479504e+00, 5.47742826e+00,
                                 5.42519447e+00, -1.45166074e+00, 2.01991521e+00, 3.93873313e+00,
                                 8.22837848e+00, 4.89401702e+00, -5.66278973e+00, 1.58617220e+00,
                                 -1.23849251e+00, -6.04649288e-01, 6.04132264e+00, 7.47588536e+00,
                                 1.82761402e+00, 3.19209639e+00, -7.58539653e-02, -6.00928508e-03,
                                 4.52902964e-01, 3.22063602e+00, 7.18613523e+00, 2.39210634e+00,
                                 -2.65743666e+00, 2.32046235e-01, 1.28079141e+00, 4.23344286e+00,
                                 6.94213238e+00, -1.15939497e+00, -5.23037351e-01, 1.54618811e+00,
                                 1.77863869e+00, 3.30139201e+00, 7.47769309e+00, -3.91312909e-01],
                                (batch_size, 1))

    y = initial_state
    eta = 0

    theta1 = np.asarray(theta1).reshape(-1, 1)

    theta2 = np.asarray(theta2).reshape(-1, 1)

    time_step = total_duration / n_timestep

    random_state = random_state or np.random

    time_series = np.empty(shape=(batch_size, n_timestep, n_obs))
    time_series[:, 0, :] = y

    for i in range(1, n_timestep):
        e = random_state.normal(0, 1, y.shape)
        eta = phi * eta + e * np.sqrt(1 - pow(phi, 2))
        params = (eta, theta1, theta2, f)

        y = runge_kutta_ode_solver(ode=_lorenz_ode, time_step=time_step, y=y, params=params)
        time_series[:, i, :] = y

    return time_series


def get_model(true_params=None, seed_obs=None, initial_state=None, n_obs=40, f=10., phi=0.984,
              total_duration=4):
    """Return a complete Lorenz model in inference task.

    This is a simplified example that achieves reasonable predictions.

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
        Initial state value of the time-series.
    n_obs : int, optional
        Number of observed variables
    f : float, optional
        Force term
    phi : float, optional
        This value is used to express stochastic forcing term. It should be configured according
        to force term and eventually impacts to the result of eta.
        More details in Wilks (2005) et al.
    total_duration : float, optional

    Returns
    -------
    m : elfi.ElfiModel

    """
    simulator = partial(forecast_lorenz, initial_state=initial_state, f=f, n_obs=n_obs, phi=phi,
                        total_duration=total_duration)

    if not true_params:
        true_params = [2.0, 0.1]

    m = elfi.ElfiModel()

    y_obs = simulator(*true_params, random_state=np.random.RandomState(seed_obs))
    sumstats = []

    elfi.Prior('uniform', 0.5, 3., model=m, name='theta1')
    elfi.Prior('uniform', 0, 0.3, model=m, name='theta2')
    elfi.Simulator(simulator, m['theta1'], m['theta2'], observed=y_obs, name='Lorenz')

    sumstats.append(elfi.Summary(mean, m['Lorenz'], name='Mean'))

    sumstats.append(elfi.Summary(var, m['Lorenz'], name='Var'))

    sumstats.append(elfi.Summary(autocov, m['Lorenz'], name='Autocov'))

    sumstats.append(elfi.Summary(cov, m['Lorenz'], name='Cov'))

    sumstats.append(elfi.Summary(xcov, m['Lorenz'], True, name='CrosscovPrev'))

    sumstats.append(elfi.Summary(xcov, m['Lorenz'], False, name='CrosscovNext'))

    elfi.Distance('euclidean', *sumstats, name='d')

    return m


def mean(x):
    """Return the mean of Y_{k}.

    Parameters
    ----------
    x : np.array of size (b, n, m) which is (batch_size, time, n_obs)

    Returns
    -------
    np.array of size (b,)
        The computed mean of statistics over time and space.

    """
    return np.mean(x, axis=(1, 2))


def var(x):
    """Return the variance of Y_{k}.

    Parameters
    ----------
    x : np.array of size (b, n, m) which is (batch_size, time, n_obs)

    Returns
    -------
    np.array of size (b,)
        The average over space of computed variance with respect to time.

    """
    return np.mean(np.var(x, axis=1), axis=1)


def cov(x):
    """Return the covariance of Y_{k} with its neighbour Y_{k+1}.

    Parameters
    ----------
    x : np.array of size (b, n, m) which is (batch_size, time, n_obs)

    Returns
    -------
    np.array of size (b,)
        The average over space of computed covariance with respect to time.

    """
    x_next = np.roll(x, -1, axis=2)
    return np.mean(np.mean((x - np.mean(x, keepdims=True, axis=1))
                           * (x_next - np.mean(x_next, keepdims=True, axis=1)),
                           axis=1), axis=1)


def xcov(x, prev=True):
    """Return the cross-covariance of Y_{k} with its neighbours from previous time step.

    Parameters
    ----------
    x : np.array of size (b, n, m) which is (batch_size, time, n_obs)
    prev : bool
        The side of previous neighbour. True for previous neighbour, False for next.

    Returns
    -------
    np.array of size (b,)
        The average over space of computed cross-covariance with respect to time.

    """
    x_lag = np.roll(x, 1, axis=2) if prev else np.roll(x, -1, axis=2)
    return np.mean((x[:, :-1, :] - np.mean(x[:, :-1, :], keepdims=True, axis=1))
                   * (x_lag[:, 1:, :] - np.mean(x_lag[:, 1:, :], keepdims=True, axis=1)),
                   axis=(1, 2))


def autocov(x):
    """Return the auto-covariance with time lag 1.

    Parameters
    ----------
    x : np.array of size (b, n, m) which is (batch_size, time, n_obs)

    Returns
    -------
    C : np.array of size (b,)
        The average over space of computed auto-covariance with respect to time.

    """
    c = np.mean((x[:, :-1, :] - np.mean(x[:, :-1, :], keepdims=True, axis=1))
                * (x[:, 1:, :] - np.mean(x[:, 1:, :], keepdims=True, axis=1)),
                axis=(1, 2))

    return c
