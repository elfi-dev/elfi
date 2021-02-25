import numpy as np

test_vec = np.arange(1225)
test_mat = np.arange(5 * 5).reshape((5, 5))

# res = np.matmul(test_vec, test_mat) / 10

# test_mat = np.array(test_vec)

print('test_mat', test_mat[:,0:1])