"""Methods for ABC diagnostics."""

import logging
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gamma

import elfi

logger = logging.getLogger(__name__)


class TwoStageSelection:
    """Perform the Two-Stage Procedure proposed by Nunes and Balding (2010).

    The rationale of the procedure is the following:
    - Initially, the combinations of the to-be-assessed summary statistics is computed;
    - In Stage 1, each summary-statistics combination is evaluated using the
    minimum-entropy algorithm;
    - In Stage 2, the minimum-entropy combination is selected,
    and the `closest' datasets are identified;
    - Further in Stage 2, for each summary-statistics combination,
    the mean root sum of squared errors (MRSSE) is calculated over all `closest datasets',
    and the minimum-MRSSE combination is chosen as the one with the optimal performance.

    Attributes
    ----------
    list_ss : array_like
        List of the to-be-assessed summary statistics.
    simulator : elfi.Simulator
        Simulator fit with the observations of the true/underlying model.
    fn_distance : str or function
        Function for calculating distance (can be a string if the function is implemented in ELFI).
    seed : int, optional

    References
    ----------
    [1] Nunes, M. A., & Balding, D. J. (2010).
    On optimal selection of summary statistics for approximate Bayesian computation.
    Statistical applications in genetics and molecular biology, 9(1).
    [2] Blum, M. G., Nunes, M. A., Prangle, D., & Sisson, S. A. (2013).
    A comparative review of dimension reduction methods in approximate Bayesian computation.
    Statistical Science, 28(2), 189-208.

    """

    def __init__(self, list_ss, simulator, fn_distance, seed=0):
        """Initialise the summary-statistics selection based on the Two-Stage Procedure.

        Parameters
        ----------
        list_ss : array_like
            List of the to-be-assessed summary statistics.
        simulator : elfi.Simulator
            Simulator fit with the observations of the true/underlying model.
        fn_distance : str or function
            Function for calculating distance
            (can be a string if the function is implemented in ELFI).
        seed : int, optional

        """
        self.list_ss = list_ss
        self.simulator = simulator
        self.fn_distance = fn_distance
        self.seed = seed

    def run(self, k, n_sim, n_acc, n_closest):
        """Run the Two-Stage Procedure.

        Parameters
        ----------
        k : int
            Parameter for the kth-nearest-neighbour search.
        n_sim : int
            Number of the total ABC-rejection simulations.
        n_acc : int
            Number of the accepted ABC-rejection simulations.
        n_closest : int
            Number of the `closest' datasets.

        Returns
        -------
        array_like
            Set of the summary statistics showing the optimal performance.

        """
        # Obtain the combinations of the summary statistics.
        sets_candidate_ss = self._create_ss_combinations(max_cardinality=k)

        # Find the summary statistics combination with the minimum entropy,
        # preserve the parameters (thetas) corresponding to the `closest' datasets.
        thetas = {}
        E_me = np.inf
        for set_ss in sets_candidate_ss:
            names_ss = [ss.__name__ for ss in set_ss]
            thetas_ss = self._obtain_accepted_thetas(set_ss, n_sim, n_acc)
            thetas[set_ss] = thetas_ss
            E_ss = self._calc_entropy(thetas_ss, n_acc, k)
            if E_ss < E_me:
                E_me = E_ss
                names_ss_me = names_ss
                thetas_closest = thetas_ss[:n_closest]
            logger.info('Combination %s shows the entropy of %f' % (names_ss, E_ss))
        logger.info('\nThe minimum entropy of %f was found in %s.\n' % (E_me, names_ss_me))

        # Find the summary-statistics combination with
        # the minimum mean root sum of squared error (MRRSE).
        MRSSE_min = np.inf
        for set_ss in sets_candidate_ss:
            names_ss = [ss.__name__ for ss in set_ss]
            MRSSE_ss = self._calc_MRSSE(set_ss, thetas_closest, thetas[set_ss])
            if MRSSE_ss < MRSSE_min:
                MRSSE_min = MRSSE_ss
                names_ss_MRSSE = names_ss
                set_ss_2stage = set_ss
            logger.info('Combination %s shows the MRSSE of %f' % (names_ss, MRSSE_ss))
        logger.info('\nThe minimum MRSSE of %f was found in %s.' % (MRSSE_min, names_ss_MRSSE))
        return set_ss_2stage

    def _create_ss_combinations(self, max_cardinality):
        """Create all combinations of the initialised summary statistics up till the maximum cardinality.

        Parameters
        ----------
        max_cardinality : int
            Cardinality limit of a summary-statistics combination.

        Returns
        -------
        array_like
            Combinations of summary statistics.

        """
        if max_cardinality > len(self.list_ss):
            max_cardinality = len(self.list_ss)

        combinations_ss = []
        for i in range(max_cardinality):
            for combination in combinations(self.list_ss, i + 1):
                combinations_ss.append(combination)
        return combinations_ss

    def _obtain_accepted_thetas(self, set_ss, n_sim, n_acc):
        """Obtain the parameters accepted by the ABC-rejection sampling.

        The sampling is performed using the initialised simulator.

        Parameters
        ----------
        set_ss : array_like
            Combination of the summary statistics used in the rejection sampling.
        n_sim : int
            Number of the iterations of the rejection sampling.
        n_acc : int
            Number of the accepted parameters.

        Returns
        -------
        array_like
            Accepted parameters.

        """
        # Initialise the distance function.
        list_ss = []
        for ss in set_ss:
            list_ss.append(elfi.Summary(ss, self.simulator))
        if isinstance(self.fn_distance, str):
            d = elfi.Distance(self.fn_distance, *list_ss)
        else:
            d = elfi.Discrepancy(self.fn_distance, *list_ss)

        # Run the simulations.
        sampler_rejection = elfi.Rejection(d, batch_size=1, seed=self.seed)
        result = sampler_rejection.sample(n_acc, n_sim=n_sim)

        # Extract the accepted parameters (thetas).
        ordered_dict_samples = result.samples
        thetas_acc = np.array(list(ordered_dict_samples.values())).T
        return thetas_acc

    def _calc_entropy(self, thetas_ss, n_acc, k):
        """Calculate the entropy as described in Nunes & Balding, 2010.

        E = log( pi^(q/2) / gamma(q/2+1) ) - digamma(k) + log(n)
            + q/n * sum_{i=1}^n( log(R_{i, k}) ), where

        R_{i, k} is the Euclidean distance from the parameter theta_i to
        its kth nearest neighbour;
        q is the dimensionality of the parameter; and
        n is the number of the accepted parameters n_acc in the rejection sampling.

        Parameters
        ----------
        thetas_ss : array_like
            Accepted parameters in the rejection sampling using
            the combination of summary statistics ss.
        n_acc : int
            Number of the accepted parameters.
        k : int
            Nearest neighbour to be searched.

        Returns
        -------
        int
            Entropy.

        """
        q = thetas_ss.shape[1]

        # Calculate the distance to the kth nearest neighbour across all accepted parameters.
        searcher_knn = cKDTree(thetas_ss)
        sum_log_dist_knn = 0
        for theta_ss in thetas_ss:
            dist_knn = searcher_knn.query(theta_ss, k=k)[0][-1]
            sum_log_dist_knn += np.log(dist_knn)

        # Calculate the entropy.
        E = np.log(np.pi**(q / 2) / gamma((q / 2) + 1)) - digamma(k) \
            + np.log(n_acc) + (q / n_acc) * sum_log_dist_knn
        return E

    def _calc_MRSSE(self, set_ss, thetas_obs, thetas_sim):
        """Calculate the mean root of squared error (MRSSE) as described in Nunes & Balding, 2010.

        MRSSE = 1/n * sum_{j=1}^n( RSSE(j) ),

        RSSE = 1/m * sum_{i=1}^m( theta_i - theta_true ), where

        n is the number of the `closest' datasets identified using
        the summary-statistics combination corresponding to the minimum entropy;
        m is the number of the accepted parameters in the rejection sampling for set_ss;
        theta_i is an instance of the parameters corresponding to set_ss; and
        theta_true is the parameters corresponding to a `closest' dataset.

        Parameters
        ----------
        set_ss : array_like
            Combination of the summary statistics used in the rejection sampling.
        thetas_obs : array_like
            Set of the parameters corresponding to the `closest' datasets.
        thetas_sim : array_like
            Set of the parameters corresponding to set_ss.

        Returns
        -------
        int
            Mean root of squared error.

        """
        RSSE_total = 0
        for theta_obs in thetas_obs:
            SSE = 0
            for theta_sim in thetas_sim:
                SSE += np.linalg.norm(theta_sim - theta_obs)**2
            RSSE = np.sqrt(SSE)
            RSSE_total += RSSE
        MRSSE = RSSE_total / len(thetas_obs)
        return MRSSE
