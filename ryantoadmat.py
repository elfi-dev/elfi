import numpy as np
import scipy.stats as ss
# from elfi.methods.parameter_inference import BSL
import matlab.engine
import elfi
import elfi.examples.toad as toad

print('typesimulate_toads2', type(simulate_toads2))
# true_alpha = 1.8
# true_delta = 45
# true_p_0 = 0.6

eng = matlab.engine.start_matlab()

true_theta = [1.8, 45, 0.6]
n_toads = 66
n_days = 63
model = 1

true_theta = matlab.double(true_theta)

true_X = eng.simulate_toads2(true_theta, n_toads, n_days, model)
eng.quit()
