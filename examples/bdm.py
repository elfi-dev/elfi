import os

import numpy as np
import elfi


"""Example implementation of the Birth-Death-Mutation (BDM)[1] model.

References
----------
[1] Tanaka, Mark M., et al. "Using approximate Bayesian computation to estimate
tuberculosis transmission parameters from genotype data."
Genetics 173.3 (2006): 1511-1520.

"""


def prepare_inputs(*inputs, **kwinputs):
    alpha, delta, tau, N = inputs
    model_name = kwinputs['meta']['model_name']
    batch_index = kwinputs['meta']['batch_index']
    submission_index = kwinputs['meta']['submission_index']

    # Prepare input parameter file
    param_array = np.row_stack(np.broadcast(alpha, delta, tau, N))
    filename = '{}_{}_{}.txt'.format(model_name, batch_index, submission_index)
    np.savetxt(filename, param_array, fmt='%.4f %.4f %.4f %d')

    # Add the filename to kwinputs
    kwinputs['filename'] = filename
    kwinputs['output_filename'] = filename[:-4] + '_out.txt'

    return inputs, kwinputs


def process_result(completed_process, *inputs, output_filename, filename, **kwinputs):
    # Read the simulations
    simulations = np.loadtxt(output_filename, dtype='int16')
    # Clean up the files
    os.remove(filename)
    os.remove(output_filename)
    return simulations


# Create an external operation callable
BDM = elfi.tools.external_operation(
    './bdm {filename} --seed {seed} --mode 1 > {output_filename}',
    prepare_inputs=prepare_inputs,
    process_result=process_result,
    stdout=False)


def T1(clusters):
    clusters = np.atleast_2d(clusters)
    return np.sum(clusters > 0, 1)/np.sum(clusters, 1)


def get_model(alpha=0.2, delta=0, tau=0.198, N=20, seed_obs=None):
    """Returns the example model used in Lintusaari et al. 2016.

    Here we infer alpha using the summary T1.

    Parameters
    ----------

    alpha : float
        birth rate
    delta : float
        death rate
    tau : float
        mutation rate
    N : int
        size of the population
    seed_obs : None, int
        Seed for the observed data generation. None gives the same data as in
        Lintusaari et al. 2016

    Returns
    -------
    m : elfi.ElfiModel
    """

    if seed_obs is None and N == 20:
        y = np.zeros(N, dtype='int16')
        data = np.array([6, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1], dtype='int16')
        y[0:len(data)] = data

    else:
        y = BDM(alpha, delta, tau, N, random_state=np.random.RandomState(seed_obs))

    m = elfi.ElfiModel(name='bdm', set_current=False)
    elfi.Prior('uniform', .005, 2, model=m, name='alpha')
    elfi.Simulator(BDM, m['alpha'], delta, tau, N, observed=y, name='BDM')
    elfi.Summary(T1, m['BDM'], name='T1')
    elfi.Distance('minkowski', m['T1'], p=1, name='d')

    m['BDM']['_uses_meta'] = True
    return m

