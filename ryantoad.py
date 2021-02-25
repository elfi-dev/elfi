from elfi.examples.toad import prepare_inputs, process_result
import elfi
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from rpy2.robjects.packages import importr
# importr('Rcpp')


command = './sim_toad {0} {1} {2} {3} {4} {seed}'

N = 66 * 63
true_params = [1.7, 35, 0.6, N] #chucked N at the end hey
alpha = 1.7
delta = 35
p0 = 0.6
batch_size = 1000
# y_obs = []

m = elfi.ElfiModel(name='toad')
elfi.Prior('uniform', 0, 1, model=m, name='alpha')
elfi.Prior('uniform', 0, 100, model=m, name='delta')
elfi.Prior('uniform', 0, 0.9, model=m, name='p0')

# Get the toad source directory
sources_path = elfi.examples.toad.get_sources_path()
print('sources_path', sources_path)
# Compile (unix-like systems)
subprocess.call(['make'], cwd=sources_path)
# Move the executable to working directory
subprocess.call(['mv', 'sources_path/toad'], shell=True) #cwd: curr??
# Create the simulator
subprocess.call(['ls'], shell=True, cwd=sources_path)




# Create an external operation callable
toad = elfi.tools.external_operation(
    './toad {filename} --seed {seed} --mode 1 > {output_filename}',
    prepare_inputs=prepare_inputs,
    process_result=process_result,
    stdout=False)



toad_node = elfi.Simulator(toad, m['alpha'], delta, p0, N)
toad_node.uses_meta = True


# print('merata', toad_node['attr_dict'])

sim_fn = toad_node['attr_dict']['_operation']
sim_results = sim_fn(n_obs=N, batch_size=batch_size, seed=123, *true_params) # TODO: MAKE AUTOMATIC n_obs, setc

print('sim_results', sim_results)

# Draw the model
# elfi.draw(m)
# plt.show(m)

res = elfi.BSL(m['_simulator'], batch_size=1000, y_obs=y_obs, n_sims=200).sample(100,
               params0=np.array(true_params))