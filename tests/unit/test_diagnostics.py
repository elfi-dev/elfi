"""Tests for ABC diagnostics."""
from functools import partial

import numpy as np

import elfi
import elfi.examples.gauss as Gauss
import elfi.examples.ma2 as MA2
from elfi.methods.diagnostics import TwoStageSelection


class TestTwoStageProcedure:
    """Tests for the Two-Stage Procedure (selection of summary statistics)."""

    @classmethod
    def setup_class(cls):
        """Refresh ELFI upon initialising the test class."""
        elfi.new_model()

    def teardown_method(self, method):
        """Refresh ELFI after the execution of the test class's each method."""
        elfi.new_model()

    def test_ma2(self, seed=0):
        """Identifying the optimal summary statistics combination following the MA2 model.

        Parameters
        ----------
        seed : int, optional

        """
        # Defining summary statistics.
        ss_mean = Gauss.ss_mean
        ss_ac_lag1 = partial(MA2.autocov, lag=1)
        ss_ac_lag1.__name__ = 'ac_lag1'
        ss_ac_lag2 = partial(MA2.autocov, lag=2)
        ss_ac_lag2.__name__ = 'ac_lag2'

        list_ss = [ss_ac_lag1, ss_ac_lag2, ss_mean]

        # Initialising the simulator.
        prior_t1 = elfi.Prior(MA2.CustomPrior1, 2, name='prior_t1')
        prior_t2 = elfi.Prior(MA2.CustomPrior2, prior_t1, 1, name='prior_t2')

        t1_true = .6
        t2_true = .2
        fn_simulator = MA2.MA2
        y_obs = fn_simulator(t1_true, t2_true, random_state=np.random.RandomState(seed))
        simulator = elfi.Simulator(fn_simulator, prior_t1, prior_t2, observed=y_obs)

        # Identifying the optimal summary statistics based on the Two-Stage procedure.
        diagnostics = TwoStageSelection(simulator, 'euclidean', list_ss=list_ss, seed=seed)
        set_ss_2stage = diagnostics.run(n_sim=100000, batch_size=10000)

        assert ss_ac_lag1 in set_ss_2stage
        assert ss_ac_lag2 in set_ss_2stage
        assert ss_mean not in set_ss_2stage
