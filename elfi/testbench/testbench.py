"""This module implements testbench-functionality to elfi"""

import logging

import numpy as np
import scipy.stats as ss

from elfi.visualization import ProgressBar

logger = logging.getLogger(__name__)

__all__ = ['Testbench', 'TestbenchMethod']


class Testbench:
    """Base class for comparing the performance of LFI-methods.
       One elfi.Model can be inferred `repetitions`-times with 
       each of the methods included in `method_list`

    Attributes
    ----------
    model : elfi.Model
        elfi.Model which is inferred.
    method_list : list
        List of elfi-infernce methods.
    repetitions : int
        How many repetitions of models is included in the testbench.
    seed : int, optional


    """

    def __init__(self,
                 model=None,
                 repetitions=1,
                 observation=None,
                 seed=None):
        """Construct the testbench object.
        Parameters
        ----------
        model : elfi.Model
            elfi.Model which is inferred. Needs to have get_model-method.
        repetitions : int
            How many repetitions of models is included in the testbench.
        observation : np.array, optional
            The observation if available
        seed : int, optional

        """

        self.model = model
        self.method_list = []
        self.repetitions = repetitions
        self._set_repetition_seeds(seed)
        self.observation = observation

    def _set_repetition_seeds(self, seed):
        """Add a new method to the testbench."""
        upper_limit = 2 ** 32 - 1
        self.obs_seeds = ss.randint(low=0, high=upper_limit).rvs(size=self.repetitions,
                                                                 random_state=seed)

    def add_method(self, new_method):
        """Add a new method to the testbench.

        Parameters
        ----------
        new_method : TestbenchMethod
            An inference method as a TestbenchMethod.

        """
        self.method_list.append(new_method)

    def run(self):
        """Run Testbench."""
        #repeated_result = {}
        method_result = []
        for testable_index, testable in enumerate(self.method_list):
            # repeated_result[testable.attributes['name']] = self._repeat_test(testable)
            method_result.append(self._repeat_test(testable))
        print(method_result)

    def _repeat_test(self, testable):
        repeated_result = []
        for i in np.arange(self.repetitions):
            model = self.model.get_model(seed_obs=self.obs_seeds[i])
            method = testable.attributes['method'](model,
                                                   **testable.attributes['method_kwargs'])

            fit_kwargs = testable.attributes['fit_kwargs']
            sampler_kwargs = testable.attributes['sample_kwargs']

            if len(fit_kwargs) > 0:
                method.fit(fit_kwargs)

            repeated_result.append(method.sample(**sampler_kwargs))

        return repeated_result


    def _compare_sample_results(self):
        """Compare results in sample-format."""

    def _retro_fitting(self):
        """Infer a problem with known parameter values."""

class TestSingleObservation(Testbench):

class TestSingleParameter(Testbench):

class TestParameterDensity(Testbench):

class TestbenchMethod:
    """Container for ParameterInference methods included in Testbench."""
    def __init__(self,
                 method,
                 method_kwargs={},
                 fit_kwargs={},
                 sample_kwargs={},
                 name=None,
                 seed=None):
        """Construct the TestbenchMethod container.

        Parameters
        ----------
        method : elfi.ParameterInference
            elfi.ParameterInfence-method which is included in Testbench.
        method_kwargs :
            Options of elfi.ParameterInference-method
        fit_kwargs :
            Options of elfi.ParameterInference.fit-method
        sample_kwargs :
            Options of elfi.ParameterInference.sample-method
        name : string, optional
            Name used the testbench
        seed : int, optional

        """
        # name = name or method.meta['name']
        self.attributes = {'method': method,
                           'method_kwargs': method_kwargs,
                           'fit_kwargs': fit_kwargs,
                           'sample_kwargs': sample_kwargs,
                           'name': name}

    def set_method_kwargs(self, **kwargs):
        """Add options for the ParameterInference contructor."""
        self.attributes['method_kwargs'] = kwargs

    def set_fit_kwargs(self, **kwargs):
        """Add options for the ParameterInference method fit()."""
        self.attributes['fit_kwargs'] = kwargs

    def set_sample_kwargs(self, **kwargs):
        """Add options for the ParameterInference method sample()."""
        self.attributes['sample_kwargs'] = kwargs

    def get_method(self):
        return self.attributes


# class GroundTruth:
#     """Base class the ground truth solution."""


# class GroundTruthParameter(GroundTruth):


# class GroundTruthPSample(GroundTruth):


# class GroundTruthObservation:


# class GroundTruthPredictedSample(GroundTruthObservation):