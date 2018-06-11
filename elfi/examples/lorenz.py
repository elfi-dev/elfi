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
        Current state of the ODE.
    params : list
        The list of parameters needed to evaluate function. In this case it is
        list of four elements - eta, theta1 and theta2.

    Returns
    -------
    dY_dt : np.array
        Rate of change of the ODE.

    """
    dY_dt = np.empty_like(y)

    eta = params[0]
    theta1 = params[1]
    theta2 = params[2]

    F = params[3]

    g = theta1 + y * theta2

    dY_dt[:, 0] = -y[:, -2] * y[:, -1] + y[:, -1] * y[:, 1] - y[:, 0] + F - g[:, 0] + eta[:, 0]

    dY_dt[:, 1] = -y[:, -1] * y[:, 0] + y[:, 0] * y[:, 2] - y[:, 1] + F - g[:, 1] + eta[:, 1]

    dY_dt[:, 2:-1] = (-y[:, :-3] * y[:, 1:-2] + y[:, 1:-2] * y[:, 3:] - y[:, 2:-1] + F - g[:, 2:-1]
                      + eta[:, 2:-1])

    dY_dt[:, -1] = (-y[:, -3] * y[:, -2] + y[:, -2] * y[:, 0] - y[:, -1] + F - g[:, -1]
                    + eta[:, -1])

    return dY_dt


def runge_kutta_ode_solver(ode, time_span, y, params):
    """4th order Runge-Kutta ODE solver.

    Carnahan, B., Luther, H. A., and Wilkes, J. O. (1969).
    Applied Numerical Methods. Wiley, New York.

    Parameters
    ----------
    ode : function
        Ordinary differential equation function
    time_span : float
        Contains the time points where the ode needs to be
        solved. The first time point corresponds to the initial value
    y : np.ndarray of dimension px1
        Current state of the time-series, corresponds to the first value of
        time_span
    params : list of parameters
        The parameters needed to evaluate the ode, i.e. eta and theta

    Returns
    -------
    np.ndarray
        Resulting state initiated at y and satisfying ode solved by this solver.

    """
    k1 = time_span * ode(y, params)

    k2 = time_span * ode(y + k1 / 2, params)

    k3 = time_span * ode(y + k2 / 2, params)

    k4 = time_span * ode(y + k3, params)

    y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y


def forecast_lorenz(theta1=None, theta2=None, f=10., phi=0.3, n_obs=40, n_timestep=160,
                    batch_size=1, initial_state=None, random_state=None, total_duration=4):
    """Forecast Lorenz model.

    Wilks, D. S. (2005). Effects of stochastic parametrizations in the
    Lorenz ’96 system. Quarterly Journal of the Royal Meteorological Society,
    131(606), 389–407.

    Parameters
    ----------
    theta1, theta2: list or numpy.ndarray
        Closure parameters. If the parameter is omitted, sampled
        from the prior.
    phi : float, optional
        Source for default value
    initial_state: numpy.ndarray, optional
        Initial state value of the time-series. The default value is zeros.
    f : float, optional
        Force term
    n_obs : int, optional
        Size of the observed 1D grid
    n_timestep : int, optional
        Number of the time step intervals
    batch_size : int, optional
    random_state : np.random.RandomState, optional
    total_duration : int, optional



    Returns
    -------
    np.ndarray
        initial_state

    """
    if not initial_state:
        initial_state = np.zeros(shape=(batch_size, n_obs))

    y_prev = y = initial_state
    eta = 0

    theta1 = np.asarray(theta1).reshape(-1, 1)

    theta2 = np.asarray(theta2).reshape(-1, 1)

    time_step = total_duration / n_timestep

    random_state = random_state or np.random

    for i in range(n_timestep):
        y_prev = y
        e = random_state.normal(0, 1, y.shape)
        eta = phi * eta + e * np.sqrt(1 - pow(phi, 2))
        params = (eta, theta1, theta2, f)
        
        y = runge_kutta_ode_solver(ode=_lorenz_ode, time_span=time_step, y=y_prev, params=params)

    y = np.stack([y, y_prev], axis=1)

    return y


def get_model(true_params=None, seed_obs=None, initial_state=None, n_obs=40, f=10., phi=0.3,
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
    n_obs : int, optional
        number of observed variables
    f : float, optional
        Force term
    phi : float, optional
    total_duration : int, optional


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

    elfi.Prior('uniform', 0.5, 3.5, model=m, name='theta1')
    elfi.Prior('uniform', 0, 0.3, model=m, name='theta2')
    elfi.Simulator(simulator, m['theta1'], m['theta2'], observed=y_obs, name='Lorenz')

    sumstats.append(elfi.Summary(mean, m['Lorenz'], name='Mean'))

    sumstats.append(elfi.Summary(var, m['Lorenz'], name='Var'))

    sumstats.append(elfi.Summary(autocov, m['Lorenz'], name='Autocov'))

    sumstats.append(elfi.Summary(cov, m['Lorenz'], name='Cov'))

    sumstats.append(elfi.Summary(xcov, m['Lorenz'], 'prev', name='CrosscovLeft'))

    sumstats.append(elfi.Summary(xcov, m['Lorenz'], 'next', name='CrosscovRight'))

    elfi.Discrepancy(chi_squared, *sumstats, name='d')

    return m


def mean(x):
    """Return the mean of Y_{k}.

    Parameters
    ----------
    x : np.array of size (b, n, m)

    Returns
    -------
    np.array of size (n,)
        The computed mean of one vector in statistics.

    """
    return np.mean(x[:, 0, :], axis=1)


def var(x):
    """Return the variance of Y_{k}.

    Parameters
    ----------
    x : np.array of size (b, n, m)

    Returns
    -------
    np.array of size (b,)
        The computed variance of two vectors in statistics.

    """
    return np.var(x[:, 0, :], axis=1)


def cov(x):
    """Return the covariance of Y_{k} with its neighbour Y_{k+1}.

    Parameters
    ----------
    x : np.array of size (b, n, m)

    Returns
    -------
    np.array of size (n,)
        The computed covariance of one vector in statistics.

    """
    x_next = np.roll(x[:, 0, :], -1, axis=1)
    return np.mean((x[:, 0, :] - np.mean(x[:, 0, :], keepdims=True, axis=1)) *
                   (x_next - np.mean(x_next, keepdims=True, axis=1)),
                   axis=1)


def xcov(x, side=None):
    """Return the cross-covariance of Y_{k} with its neighbours from previous time steps.

    Parameters
    ----------
    x : np.array of size (b, n, m)

    Returns
    -------
    np.array of size (n,)
        The computed cross-covariance of two vectors in statistics.

    """
    # cross-co-variance with left neighbour Y_{k-1}
    if side == 'prev':
        x_prev = np.roll(x[:, 1, :], 1, axis=1)
        return np.mean((x[:, 0, :] - np.mean(x[:, 0, :], keepdims=True, axis=1)) *
                       (x_prev - np.mean(x_prev, keepdims=True, axis=1)),
                       axis=1)

    # cross-co-variance with right neighbour Y_{k+1}
    elif side == 'next':
        x_next = np.roll(x[:, 1, :], -1, axis=1)
        return np.mean((x[:, 0, :] - np.mean(x[:, 0, :], keepdims=True, axis=1)) *
                       (x_next - np.mean(x_next, keepdims=True, axis=1)),
                       axis=1)


def autocov(x):
    """Return the autocovariance.

    Realizations are in rows.

    Parameters
    ----------
    x : np.array of size (b, n, m)
    lag : int, optional

    Returns
    -------
    C : np.array of size (n,)
        The computed auto-covariance of two vectors in statistics.

    """
    c = np.mean((x[:, 0, :] - np.mean(x[:, 0, :], keepdims=True, axis=1)) *
                (x[:, 1, :] - np.mean(x[:, 1, :], keepdims=True, axis=1)),
                axis=1)

    return c


def chi_squared(*simulated, observed):
    """Return Chi squared goodness of fit. Adjusts for differences in magnitude between dimensions.

    Parameters
    ----------
    observed : np.arrays
    simulated : tuple of np.arrays
        The tuple of all summary statistics

    Returns
    -------
    c : ndarray
        The calculated cost function

    """
    simulated = np.column_stack(simulated)
    observed = np.column_stack(observed)

    d = np.sum((simulated - observed) ** 2. / observed, axis=1)

    return d
