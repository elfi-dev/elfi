"""Tests for ABC diagnostics."""
from functools import partial

import numpy as np

import elfi
import elfi.examples.bignk as BiGNK
import elfi.examples.gauss as Gauss
import elfi.examples.gnk as GNK
from elfi.examples.gnk import euclidean_multiss
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
        diagnostics = TwoStageSelection(list_ss, simulator, 'euclidean', seed)
        set_ss_2stage = diagnostics.run(k=4, n_sim=1000, n_acc=100, n_closest=20)

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
        # simulator = elfi.Simulator(fn_simulator, observed=y_obs)
        simulator = elfi.Simulator(fn_simulator, *priors, observed=y_obs)

        # Identifying the optimal summary statistics based on the Two-Stage procedure.
        diagnostics = TwoStageSelection(list_ss, simulator, 'euclidean', seed)
        diagnostics.run(k=4, n_sim=1000, n_acc=100, n_closest=20)

    def test_bignk(self, seed=0):
        """Identifying the optimal summary statistics combination following the bivariate-g-and-k model.

        Testing the Two-Stage Procedure's implementation for 2-D data.

        Parameters
        ----------
        seed : int, optional

        """
        # Defining summary statistics.
        ss_order = GNK.ss_order
        ss_robust = GNK.ss_robust
        ss_octile = GNK.ss_octile

        list_ss = [ss_order, ss_robust, ss_octile]

        # Initialising the simulator.
        true_params = [3, 4, 1, 0.5, 1, 2, .5, .4, 0.6]
        priors = []
        priors.append(elfi.Prior('uniform', 0, 5, name='A1'))
        priors.append(elfi.Prior('uniform', 0, 5, name='A2'))
        priors.append(elfi.Prior('uniform', 0, 5, name='B1'))
        priors.append(elfi.Prior('uniform', 0, 5, name='B2'))
        priors.append(elfi.Prior('uniform', -5, 10, name='g1'))
        priors.append(elfi.Prior('uniform', -5, 10, name='g2'))
        priors.append(elfi.Prior('uniform', -.5, 5.5, name='k1'))
        priors.append(elfi.Prior('uniform', -.5, 5.5, name='k2'))
        EPS = np.finfo(float).eps
        priors.append(elfi.Prior('uniform', -1 + EPS, 2 - 2 * EPS, name='rho'))

        fn_simulator = BiGNK.BiGNK
        y_obs = fn_simulator(*true_params, random_state=np.random.RandomState(seed))
        simulator = elfi.Simulator(fn_simulator, *priors, observed=y_obs)

        # Identifying the optimal summary statistics based on the Two-Stage procedure.
        diagnostics = TwoStageSelection(list_ss, simulator, euclidean_multiss, seed)
        diagnostics.run(k=4, n_sim=1000, n_acc=100, n_closest=20)

    def test_gnk(self, seed=0):
        """Identifying the optimal summary statistics combination following the g-and-k model.

        Testing the Two-Stage Procedure's implementation for 1-D data.

        Parameters
        ----------
        seed : int, optional

        """
        # Defining summary statistics.
        ss_order = GNK.ss_order
        ss_robust = GNK.ss_robust
        ss_octile = GNK.ss_octile

        list_ss = [ss_order, ss_robust, ss_octile]

        # Initialising the simulator.
        true_params = [3, 1, 2, .5]
        priors = []
        priors.append(elfi.Prior('uniform', 0, 10, name='A'))
        priors.append(elfi.Prior('uniform', 0, 10, name='B'))
        priors.append(elfi.Prior('uniform', 0, 10, name='g'))
        priors.append(elfi.Prior('uniform', 0, 10, name='k'))

        fn_simulator = GNK.GNK
        y_obs = fn_simulator(*true_params, random_state=np.random.RandomState(seed))
        simulator = elfi.Simulator(fn_simulator, *priors, observed=y_obs)

        # Identifying the optimal summary statistics based on the Two-Stage procedure.
        diagnostics = TwoStageSelection(list_ss, simulator, euclidean_multiss, seed)
        diagnostics.run(k=4, n_sim=1000, n_acc=100, n_closest=20)
