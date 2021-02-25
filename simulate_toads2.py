import numpy as np
import scipy.stats as ss

def simulate_toads2(alpha, gamma, p0, n_toads, n_days, model):
    """simulate toad locations for each toad and day
    theta: model parameters, alpha, gamma and p0 (the probability of 
           returning to a previous refuge)
    n_toads: the number of individual toads
    n_days: the number of days for simulation
    model: indicator for model version in the paper Marchand, et al (2017),
           1 stands for random return
           2 stands for nearest return

    X: a ndays by ntoads matrix contains toads location for each toad
    """

    # alpha = theta[0]
    # gamma = theta[1]
    # p0 = theta[2]


    X = np.zeros((n_days, n_toads))

    for i in range(n_days - 1):
        i += 1
        if (model == 1): #  random return
            ind = np.random.uniform(0, 1, n_toads) >= p0
            non_ind = np.invert(ind)
            # print('alpha', alpha, 'gamma', gamma)
            delta_x = np.transpose(ss.levy_stable.rvs(alpha, beta=0, scale=gamma, size=np.sum(ind)))
            # print('ind', ind)
            X[i, ind] = X[i-1, ind] + delta_x

            ind_refuge = np.random.choice(int(i), size=n_toads - np.sum(ind))
            # idx = np.ravel_multi_index((ind_refuge, np.argwhere(non_ind)), X.shape)
            non_ind_idx = np.argwhere(non_ind).flatten()
            # print('ind_refuge', ind_refuge.shape)
            # print('non_ind_idx', non_ind_idx.shape)
            # print('np.ix_(ind_refuge,non_ind_idx)', np.ix_(ind_refuge,non_ind_idx))
            X[i, non_ind_idx] = X[ind_refuge, non_ind_idx]

    return X
        # return to the closest refuge site with probability p0
        # not implemented here
        # else if model == 2: