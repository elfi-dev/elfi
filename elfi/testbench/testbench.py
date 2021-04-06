"""This module implements testbench-functionality to elfi"""

import logging

import numpy as np
import scipy.stats as ss

from elfi.visualization.visualization import ProgressBar

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
                 observations=None,
                 reference_parameter=None,
                 reference_posterior=None,
                 progress_bar=True,
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
        reference_parameter : np.array, optional
            True parameter value if available
        reference_posterior : 
        progress_bar : boolean
            Indicate whether to display testbench progressbar.
        seed : int, optional

        """

        self.model = model
        self.method_list = []
        self.repetitions = repetitions
        self._set_repetition_seeds(seed)
        self.observations = observations
        self.reference_parameter = reference_parameter
        self.reference_posterior = reference_posterior
        self.simulator_name = list(model.observed)[0]
        self._resolve_test_type()
        self.progress_bar = ProgressBar(prefix='Progress', suffix='Complete',
                                        decimals=1, length=50, fill='=')

    def _set_repetition_seeds(self, seed):
        """Fix a seed for each of the repeated instances."""
        upper_limit = 2 ** 32 - 1
        self.obs_seeds = ss.randint(
            low=0, high=upper_limit).rvs(size=self.repetitions,
                                         random_state=seed)

    def _resolve_test_type(self):
        self._set_default_test_type()
        self._resolve_reference_parameters()
        self._resolve_observations()

    def _set_default_test_type(self):
        self.description = {
            'observations_available': self.observations is not None,
            'reference_parameters_available': self.reference_parameter is not None,
            'reference_posterior_available': self.reference_posterior is not None
            }

    def _resolve_reference_parameters(self):
        if self.description['reference_parameters_available']:
            self.reference_parameter = np.repeat(
                self.reference_parameter,
                repeats=self.repetitions,
                axis=0)
        else:
            self.reference_parameter = self.model.generate(
                batch_size=self.repetitions,
                outputs=self.model.parameter_names,
                seed=None)

    def _resolve_observations(self):
        if self.description['observations_available']:
            self.observations = np.repeat(
                self.observations,
                repeats=self.repetitions,
                axis=0)
        else:
            self.observations = self.model.generate(
                batch_size=self.repetitions,
                with_values=self.true_parameter,
                seed=None)

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
        method_result = []
        for testable_index, testable in enumerate(self.method_list):
            self.progress_bar.reinit_progressbar(reinit_msg=testable.attributes['method'])
            # repeated_result[testable.attributes['name']] = self._repeat_test(testable)
            method_result.append(self._repeat_method_test(testable))
        print(method_result)

    def _repeat_method_test(self, testable):
        repeated_result = []
        model_instance = self.model.copy
        for i in np.arange(self.repetitions):
            self.progress_bar.update_progressbar(i + 1, self.repetitions)
            model_instance.observed[self.simulator_name] = self.observations[i]
            # model = self.model.get_model(seed_obs=self.obs_seeds[i])
            method = testable.attributes['method'](
                model_instance, **testable.attributes['method_kwargs'])

            fit_kwargs = testable.attributes['fit_kwargs']
            sampler_kwargs = testable.attributes['sample_kwargs']

            if len(fit_kwargs) > 0:
                method.fit(fit_kwargs)

            repeated_result.append(method.sample(**sampler_kwargs))

        return self._combine_method_results(testable.attributes['name'],
                                            repeated_result)

    def _combine_method_results(self, name, results):
        result_dictionary = {
            'method': name,
            'results': results,
            'observations': self.observations,
            'reference_parameters': self.reference_parameters,
            'reference_posterior': self.reference_posterior
        }
        return result_dictionary

    def _compare_sample_results(self):
        """Compare results in sample-format."""

    def _retrodiction(self):
        """Infer a problem with known parameter values."""


class TestSingleObservation(Testbench):
    def __init__(self):
        super(TestSingleObservation, self).__init__()


class TestSingleParameter(Testbench):
    def __init__(self):
        super(TestSingleParameter, self).__init__()


class TestParameterDensity(Testbench):
    def __init__(self):
        super(TestParameterDensity, self).__init__()


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
        name = name or method.__name__
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