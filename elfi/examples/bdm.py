"""The model used in Lintusaari et al. 2016 with summary statistic T1.

References
----------
- Jarno Lintusaari, Michael U. Gutmann, Ritabrata Dutta, Samuel Kaski, Jukka Corander;
  Fundamentals and Recent Developments in Approximate Bayesian Computation.
  Syst Biol 2017; 66 (1): e66-e82. doi: 10.1093/sysbio/syw077

"""

import os
import warnings

import numpy as np

import elfi


def prepare_inputs(*inputs, **kwinputs):
    """Prepare the inputs for the simulator.

    The signature follows that given in `elfi.tools.external_operation`. This function
    appends kwinputs with unique and descriptive filenames and writes an input file for
    the bdm executable.
    """
    alpha, delta, tau, N = inputs
    meta = kwinputs['meta']

    # Organize the parameters to an array. The broadcasting works nicely with constant
    # arguments.
    param_array = np.row_stack(np.broadcast(alpha, delta, tau, N))

    # Prepare a unique filename for parallel settings
    filename = '{model_name}_{batch_index}_{submission_index}.txt'.format(**meta)
    np.savetxt(filename, param_array, fmt='%.4f %.4f %.4f %d')

    # Add the filenames to kwinputs
    kwinputs['filename'] = filename
    kwinputs['output_filename'] = filename[:-4] + '_out.txt'

    # Return new inputs that the command will receive
    return inputs, kwinputs


def process_result(completed_process, *inputs, **kwinputs):
    """Process the result of the BDM simulation.

    The signature follows that given in `elfi.tools.external_operation`.
    """
    output_filename = kwinputs['output_filename']

    # Read the simulations from the file.
    simulations = np.loadtxt(output_filename, dtype='int16')

    # Clean up the files after reading the data in
    os.remove(kwinputs['filename'])
    os.remove(output_filename)

    # This will be passed to ELFI as the result of the command
    return simulations


# Create an external operation callable
BDM = elfi.tools.external_operation(
    './bdm {filename} --seed {seed} --mode 1 > {output_filename}',
    prepare_inputs=prepare_inputs,
    process_result=process_result,
    stdout=False)


def T1(clusters):
    """Summary statistic for BDM."""
    clusters = np.atleast_2d(clusters)
    return np.sum(clusters > 0, 1) / np.sum(clusters, 1)


def T2(clusters, n=20):
    """Another summary statistic for BDM."""
    clusters = np.atleast_2d(clusters)
    return 1 - np.sum((clusters / n)**2, axis=1)


def get_sources_path():
    """Return the path to the C++ source code."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cpp')


def get_model(alpha=0.2, delta=0, tau=0.198, N=20, seed_obs=None):
    """Return the example model used in Lintusaari et al. 2016.

    Here we infer alpha using the summary statistic T1. We expect the executable `bdm` be
    available in the working directory.

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

    m = elfi.ElfiModel(name='bdm')
    elfi.Prior('uniform', .005, 2, model=m, name='alpha')
    elfi.Simulator(BDM, m['alpha'], delta, tau, N, observed=y, name='BDM')
    elfi.Summary(T1, m['BDM'], name='T1')
    elfi.Distance('minkowski', m['T1'], p=1, name='d')

    m['BDM'].uses_meta = True

    # Warn the user if the executable is not present
    if not os.path.isfile('bdm') and not os.path.isfile('bdm.exe'):
        cpp_path = get_sources_path()
        warnings.warn("This model uses an external simulator `bdm` implemented in C++ "
                      "that needs to be compiled and copied to your working directory. "
                      "We could not find it from your current working directory. Please"
                      "copy the folder `{}` to your working directory "
                      "and compile the source.".format(cpp_path), RuntimeWarning)

    return m
