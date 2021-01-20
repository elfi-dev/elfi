"""The model used in...
References...
"""
import os
import warnings

import numpy as np

import elfi

command = './toad {filename} --seed {seed} --mode 1 > {output_filename}'

# Function to prepare the inputs for the simulator. Create file
# names and write an input file
def prepare_inputs(*inputs, **kwinputs):
    alpha, delta, p0, N = inputs
    # meta = kwinputs['meta']
    # print('meta', meta)
    # Organise the params to an array. The broadcasting works nicely with constant args here
    param_array = np.row_stack(np.broadcast(alpha, delta, p0, N))

    # Prepare a unique filename for parallel settings
    filename = 'tempfilename.txt'
    # filename = '{model_name}_{batch_index}_{submission_index}.txt'.format(**meta)
    np.savetxt(filename, param_array, fmt='%.4f %.4f %.4f %d')

    # Add the filenames to kwinputs
    kwinputs['filename'] = filename
    kwinputs['output_filename'] = filename[:-4] + '_out.txt'

    # Return new inputs that the command will receive
    return inputs, kwinputs

# Function to process the result of the simulation
def process_result(completed_process, *inputs, **kwinputs):
    output_filename = kwinputs['output_filename']

    # Read the simulations from the file
    simulations = np.loadtxt(output_filename, dtype='int16')

    # Clean up the files after reading the data in
    os.remove(kwinputs['filename'])
    os.remove(output_filename)

    # This will be passed to ELFI as the result of the command
    return simulations


# Summary Stats
def get_sources_path():
    """Return the path to the C++ source code."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cpp/toad')

def get_model(alpha=1.7, delta=35, p0=0.6, N=300, seed_obs=None):
    pass
