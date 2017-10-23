"""Tests for ABC diagnostics."""
from functools import partial

import numpy as np

import elfi
import elfi.examples.gauss as Gauss
import elfi.examples.ma2 as MA2
from elfi.examples.ma2 import CustomPrior1, CustomPrior2
from elfi.methods.diagnostics import TwoStageSelection


class TestTwoStageProcedure:
    """Tests for the Two-Stage Procedure."""

    @classmethod
    def setup_class(cls):
        """Refresh ELFI upon initialising the test class."""
        elfi.new_model()

    def teardown_method(self, method):
        """Refresh ELFI after the execution of the test class's each method."""
        elfi.new_model()

    def test_fixedvar_gaussian(self, seed=0):
        """Identifying the optimal summary statistics combination following the Gaussian-noise model.

        Testing the Two-Stage Procedure's implementation for 1-D parameters.

        Parameters
        ----------
        seed : int, optional

        """
        # Defining summary statistics.
        def ss_uninformative(x):
            return 1
        ss_mean = Gauss.ss_mean
        ss_var = Gauss.ss_var

        list_ss = [ss_mean, ss_var, ss_uninformative]

        # Initialising the simulator.
        mean_true = 5
        cov_matrix = [1]
        prior = elfi.Prior('uniform', 2.5, 5, name='mu')

        fn_simulator = partial(Gauss.gauss_nd_mean, cov_matrix=cov_matrix)
        y_obs = fn_simulator(mean_true, cov_matrix=cov_matrix,
                             random_state=np.random.RandomState(seed))
        simulator = elfi.Simulator(fn_simulator, prior, observed=y_obs)

        # Identifying the optimal summary statistics based on the Two-Stage procedure.
        diagnostics = TwoStageSelection(simulator, 'euclidean', list_ss=list_ss, seed=seed)
        set_ss_2stage = diagnostics.run(n_sim=2000)

        assert(ss_mean in set_ss_2stage and ss_var not in set_ss_2stage and
               ss_uninformative not in set_ss_2stage)

    def test_gaussian(self, seed=0):
        """Identifying the optimal summary-statistics combination following the Gaussian-noise model.

        Testing the Two-Stage Procedure's implementation for 2-D parameters.

        Parameters
        ----------
        seed : int, optional

        """
        # Defining summary statistics.
        def ss_uninformative(x):
            return 1
        ss_mean = Gauss.ss_mean
        ss_var = Gauss.ss_var

        list_ss = [ss_mean, ss_var, ss_uninformative]

        # Initialising the simulator.
        mean_true = 5
        std_true = 1
        priors = []
        priors.append(elfi.Prior('uniform', 0, 10, name='mu'))
        priors.append(elfi.Prior('truncnorm', 0, 4, name='sigma'))

        fn_simulator = Gauss.gauss
        y_obs = fn_simulator(mean_true, std_true, random_state=np.random.RandomState(seed))
        simulator = elfi.Simulator(fn_simulator, *priors, observed=y_obs)

        # Identifying the optimal summary statistics based on the Two-Stage procedure.
        diagnostics = TwoStageSelection(simulator, 'euclidean', list_ss=list_ss, seed=seed)
        set_ss_2stage = diagnostics.run(n_sim=2000)

        assert(ss_uninformative not in set_ss_2stage)

    def test_ma2(self, seed=0):
        """Identifying the optimal summary statistics combination following the MA2 model.

        Parameters
        ----------
        seed : int, optional

        """
        # Defining summary statistics.
        ss_mean = Gauss.ss_mean
        ss_var = Gauss.ss_var
        ss_ac_lag1 = partial(MA2.autocov, lag=1)
        ss_ac_lag1.__name__ = 'ac_lag1'
        ss_ac_lag2 = partial(MA2.autocov, lag=2)
        ss_ac_lag2.__name__ = 'ac_lag2'

        list_ss = [ss_ac_lag1, ss_ac_lag2, ss_mean, ss_var]

        # Initialising the simulator.
        prior_t1 = elfi.Prior(CustomPrior1, 2, name='prior_t1')
        prior_t2 = elfi.Prior(CustomPrior2, prior_t1, 1, name='prior_t2')

        t1_true = .6
        t2_true = .2
        fn_simulator = MA2.MA2
        y_obs = fn_simulator(t1_true, t2_true, random_state=np.random.RandomState(seed))
        simulator = elfi.Simulator(fn_simulator, prior_t1, prior_t2, observed=y_obs)

        # Identifying the optimal summary statistics based on the Two-Stage procedure.
        diagnostics = TwoStageSelection(simulator, 'euclidean', list_ss=list_ss, seed=seed)
        set_ss_2stage = diagnostics.run(n_sim=2000)

        assert(ss_ac_lag1 in set_ss_2stage and ss_ac_lag2 in set_ss_2stage)
