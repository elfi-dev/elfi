"""Methods for ABC diagnostics."""

import logging
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gamma

import elfi

logger = logging.getLogger(__name__)


class TwoStageSelection:
    """Perform the summary-statistics selection proposed by Nunes and Balding (2010).

    The user can provide a list of summary statistics, list_ss, and let ELFI to combine them,
    or provide the already prepared combinations as prepared_ss.

    The rationale of the procedure is the following:
    - First, the module computes or accepts the combinations of the candidate summary statistics;
    - In Stage 1, each summary-statistics combination is evaluated using the
    minimum-entropy algorithm;
    - In Stage 2, the minimum-entropy combination is selected,
    and the `closest' datasets are identified;
    - Further in Stage 2, for each summary-statistics combination,
    the mean root sum of squared errors (MRSSE) is calculated over all `closest datasets',
    and the minimum-MRSSE combination is chosen as the one with the optimal performance.

    Notes
    -----
    Future extension: include different distance functions in the summary-statistics combinations.

    Attributes
    ----------
    simulator : elfi.Simulator
        Simulator initialised with the observations and the priors.
    fn_distance : str or callable function
        Function for calculating distance (can be a string if the function is implemented in ELFI).
    seed : int, optional
    max_cardinality : int, optional
        Maximum cardinality of a candidate summary-statistics combination.
    list_ss : array_like of callable functions, optional
        List of uncombined candidate summary statistics.
    prepared_ss : array_like of callable functions, optional
        List of combined candidate summary statistics.

    References
    ----------
    [1] Nunes, M. A., & Balding, D. J. (2010).
    On optimal selection of summary statistics for approximate Bayesian computation.
    Statistical applications in genetics and molecular biology, 9(1).
    [2] Blum, M. G., Nunes, M. A., Prangle, D., & Sisson, S. A. (2013).
    A comparative review of dimension reduction methods in approximate Bayesian computation.
    Statistical Science, 28(2), 189-208.

    """

    def __init__(self, simulator, fn_distance, seed=0, max_cardinality=4,
                 list_ss=None, prepared_ss=None):
        """Initialise the summary-statistics selection for the Two-Stage Procedure.

        Parameters
        ----------
        simulator : elfi.Simulator
            Simulator initialised with the observations and the priors.
        fn_distance : str or callable function
            Distance metric (can be a string if the metric is implemented in ELFI).
        seed : int, optional
        max_cardinality : int, optional
            Maximum cardinality of a candidate summary-statistics combination.
        list_ss : array_like of callable functions, optional
            List of uncombined candidate summary statistics.
        prepared_ss : array_like of callable functions, optional
            List of combined candidate summary statistics.

        """
        if list_ss is None and prepared_ss is None:
            raise ValueError('No summary statistics to assess.')

        self.simulator = simulator
        self.fn_distance = fn_distance
        self.seed = seed
        if prepared_ss is not None:
            self.ss_candidates = prepared_ss
        else:
            self.ss_candidates = self._combine_ss(list_ss, max_cardinality=max_cardinality)
        # Initialising an output pool as the rejection sampling will be used several times.
        names_prior = [prior.name for prior in simulator.parents]
        self.pool = elfi.OutputPool(names_prior)

    def _combine_ss(self, list_ss, max_cardinality):
        """Create all combinations of the initialised summary statistics up till the maximum cardinality.

        Parameters
        ----------
        list_ss : array_like of callable functions
            List of uncombined candidate summary statistics.
        max_cardinality : int
            Maximum cardinality of a candidate summary-statistics combination.

        Returns
        -------
        array_like
            Combinations of candidate summary statistics.

        """
        if max_cardinality > len(list_ss):
            max_cardinality = len(list_ss)

        # Combine the candidate summary statistics.
        combinations_ss = []
        for i in range(max_cardinality):
            for combination in combinations(list_ss, i + 1):
                combinations_ss.append(combination)
        return combinations_ss

    def run(self, n_sim, n_acc, n_closest, k=4):
        """Run the Two-Stage Procedure.

        Parameters
        ----------
        n_sim : int
            Number of the total ABC-rejection simulations.
        n_acc : int
            Number of the accepted ABC-rejection simulations.
        n_closest : int
            Number of the `closest' datasets.
        k : int, optional
            Parameter for the kth-nearest-neighbour search.

        Returns
        -------
        array_like
            Summary-statistics combination showing the optimal performance.

        """
        # Find the summary-statistics combination with the minimum entropy, and
        # preserve the parameters (thetas) corresponding to the `closest' datasets.
        thetas = {}
        E_me = np.inf
        for set_ss in self.ss_candidates:
            names_ss = [ss.__name__ for ss in set_ss]
            thetas_ss = self._obtain_accepted_thetas(set_ss, n_sim, n_acc)
            thetas[set_ss] = thetas_ss
            E_ss = self._calc_entropy(thetas_ss, n_acc, k)
            # If equal, dismiss the combination which contains uninformative summary statistics.
            if (E_ss == E_me and (len(names_ss_me) > len(names_ss))) or E_ss < E_me:
                E_me = E_ss
                names_ss_me = names_ss
                thetas_closest = thetas_ss[:n_closest]
            logger.info('Combination %s shows the entropy of %f' % (names_ss, E_ss))
        logger.info('\nThe minimum entropy of %f was found in %s.\n' % (E_me, names_ss_me))

        # Find the summary-statistics combination with
        # the minimum mean root sum of squared error (MRRSE).
        MRSSE_min = np.inf
        for set_ss in self.ss_candidates:
            names_ss = [ss.__name__ for ss in set_ss]
            MRSSE_ss = self._calc_MRSSE(set_ss, thetas_closest, thetas[set_ss])
            # If equal, dismiss the combination which contains uninformative summary statistics.
            if (MRSSE_ss == MRSSE_min and (len(names_ss_MRSSE) > len(names_ss))) \
                    or MRSSE_ss < MRSSE_min:
                MRSSE_min = MRSSE_ss
                names_ss_MRSSE = names_ss
                set_ss_2stage = set_ss
            logger.info('Combination %s shows the MRSSE of %f' % (names_ss, MRSSE_ss))
        logger.info('\nThe minimum MRSSE of %f was found in %s.' % (MRSSE_min, names_ss_MRSSE))
        return set_ss_2stage

    def _obtain_accepted_thetas(self, set_ss, n_sim, n_acc):
        """Perform the ABC-rejection sampling and identify `closest' parameters.

        The sampling is performed using the initialised simulator.

        Parameters
        ----------
        set_ss : array_like
            Summary-statistics combination to be used in the rejection sampling.
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
        m = self.simulator.model.copy()
        list_ss = []
        for ss in set_ss:
            list_ss.append(elfi.Summary(ss, m[self.simulator.name], model=m))
        if isinstance(self.fn_distance, str):
            d = elfi.Distance(self.fn_distance, *list_ss)
        else:
            d = elfi.Discrepancy(self.fn_distance, *list_ss)

        # Run the simulations.
        sampler_rejection = elfi.Rejection(d, batch_size=1, seed=self.seed, pool=self.pool)
        result = sampler_rejection.sample(n_acc, n_sim=n_sim)

        # Extract the accepted parameters.
        thetas_acc = result.samples_array
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
            Parameters accepted upon the rejection sampling using
            the summary-statistics combination ss.
        n_acc : int
            Number of the accepted parameters.
        k : int
            Nearest neighbour to be searched.

        Returns
        -------
        float
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
            Summary-statistics combination used in the rejection sampling.
        thetas_obs : array_like
            List of parameters corresponding to the `closest' datasets.
        thetas_sim : array_like
            Parameters corresponding to set_ss.

        Returns
        -------
        float
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
