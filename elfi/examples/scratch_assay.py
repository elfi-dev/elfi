"""
Example implementation of scratch assay simulation.

References
----------
Johnston et al (2014) Interpreting scratch assays using pair density dynamics and approximate
Bayesian computation. Open Biol. 4: 140097. https://doi.org/10.1098/rsob.140097

Price et al (2018) Bayesian synthetic likelihood. J. Computational and Graphical Statistics,
27(1), 1-11. https://doi.org/10.1080/10618600.2017.1302882

"""

import numpy as np

import elfi


def _random_init(nrows, ncols, ncell, nrows_init, random_state=None):

    random_state = random_state or np.random
    init = np.zeros(nrows*ncols)
    init[:ncell] = np.ones(ncell)
    init[:nrows_init*ncols] = random_state.permutation(init[:nrows_init*ncols])
    return init.reshape(nrows, ncols)


def _random_move(coords, nrows, ncols, random_state=None):

    random_state = random_state or np.random
    move_set = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    prop_coords = np.array(coords) + move_set[random_state.choice(4)]
    prop_coords = np.minimum(np.maximum(prop_coords, 0), [nrows-1, ncols-1])
    return prop_coords


def cell_sim(pm, pp, init_arr=None, init_params=None, obs_period=12, obs_interval=1/12, tau=1/24,
             random_state=None):
    """Generate cell array observations as proposed by Johnston et al (2014).

    Parameters
    ----------
    pm : float
        Motility event probability.
    pp : float
        Proliferation event probability.
    init_arr : np.ndarray, optional
        Initial cell locations as binary array.
    init_params : list, optional
        Parameters for random initialisation. Not used if init_arr is provided.
    obs_period : float, optional
        Observation period.
    obs_interval : float, optional
        Time between observations.
    tau : float, optional
        Time step.
    random_state : np.random.RandomState, optional

    Returns
    -------
    np.ndarray (nrows, ncols, num_obs)

    """
    random_state = random_state or np.random

    if init_arr is None:
        init_params = init_params or [27, 36, 100, 10]
        cell_arr = _random_init(*init_params, random_state=random_state)
    else:
        cell_arr = np.copy(init_arr)

    nrows, ncols = cell_arr.shape
    num_iter = int(obs_period/tau)
    obs_interval = int(obs_interval/tau)
    num_obs = int(num_iter/obs_interval)
    obs_arr = np.ones((num_obs+1, nrows, ncols))
    obs_arr[0] = np.copy(cell_arr)

    for iteration in range(num_iter):

        # update cell count and coordinates

        num_cells = int(np.sum(cell_arr))
        cell_coords = np.transpose(np.array(np.where(cell_arr)))

        if num_cells == nrows * ncols:
            continue

        # motility events

        candidates = random_state.choice(num_cells, size=num_cells)
        p = random_state.uniform(size=num_cells)
        candidates = candidates[p < pm]

        for cell in candidates:
            coords = _random_move(cell_coords[cell], nrows, ncols, random_state)
            if cell_arr[coords[0], coords[1]] == 0:
                # move
                cell_arr[cell_coords[cell][0], cell_coords[cell][1]] = 0
                cell_arr[coords[0], coords[1]] = 1
                # update coordinate list
                cell_coords[cell] = coords

        # proliferation events

        candidates = random_state.choice(num_cells, size=num_cells)
        p = random_state.uniform(size=num_cells)
        candidates = candidates[p < pp]

        for cell in candidates:
            coords = _random_move(cell_coords[cell], nrows, ncols, random_state)
            cell_arr[coords[0], coords[1]] = 1   # increases cell count if coords is unoccupied

        # observations

        if (iteration + 1) % obs_interval == 0:
            obs_arr[int((iteration + 1)/obs_interval)] = np.copy(cell_arr)

    return np.transpose(obs_arr, (1, 2, 0))


def cell_summaries(x):
    """Calculate summary statistics proposed by Price et al (2018).

    Parameters
    ----------
    x : np.ndarray
        Simulated/observed data in shape (batch_size, nrows, ncols, num_obs).

    Returns
    -------
    np.ndarray (batch_size, num_obs+1)

    """
    # 1. mismatch between consecutive observations
    ds = np.sum(np.abs((x[:, :, :, : -1] - x[:, :, :, 1:])), axis=(1, 2))

    # 2. final cell count
    count = np.sum(x[:, :, :, -1], axis=(1, 2))[:, None]

    return np.concatenate((ds, count), axis=1)


def get_model(true_params=None, init_arr=None, init_params=None, seed_obs=None):
    """
    Return complete scratch assay model.

    Parameters
    ----------
    true_params : list, optional
        Parameters with which the observed data is generated.
    init_arr : np.ndarray, optional
        Initial cell locations as binary array.
    init_params : list, optional
        Parameters for random initialisation. Not used if init_arr is provided.
    seed_obs : int, optional
        Seed for the observed data generation.

    Returns
    -------
    elfi.ElfiModel

    """
    if true_params is None:
        true_params = [0.25, 0.002]

    m = elfi.ElfiModel()

    # priors
    elfi.Prior('uniform', 0, 1, model=m, name='pm')
    elfi.Prior('uniform', 0, 1, model=m, name='pp')

    # observed data
    random_state = np.random.RandomState(seed_obs)
    obs = cell_sim(*true_params, init_arr, init_params, random_state=random_state)

    # simulator
    cell_sim_vector = elfi.tools.vectorize(cell_sim, constants=(2,))
    init_arr = obs[:, :, 0]
    obs = obs[None, :]
    elfi.Simulator(cell_sim_vector, m['pm'], m['pp'], init_arr, name='sim', observed=obs)

    # summaries
    elfi.Summary(cell_summaries, m['sim'], name='sums')

    # distance
    num_ds = m['sums'].observed.size - 1
    num_init = np.sum(init_arr)
    weis = np.concatenate((np.ones(num_ds)/num_ds, np.array([1])))/num_init**2
    elfi.Distance('euclidean', m['sums'], w=weis, name='d')

    return m
