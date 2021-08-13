# import numpy as np
# import math
# import matlab.engine

# # ground-true params
# om = 0.3
# ok = 0
# w0 = -1
# wa = 0
# h0 = 0.7

# log_om = math.log(om)
# log_h0 = math.log(h0)

# thetas = [log_om, ok, w0, wa, log_h0]

# n = 10e+4
# n_bin = 20
# num_sim = 1000

# eng = matlab.engine.start_matlab()

# thetas = matlab.double(thetas)

# sim_summ = eng.astroSL_simsummaries(thetas, n, n_bin, num_sim)
# print('sim_summ', sim_summ)

# eng.quit()

# # sim_summ = 