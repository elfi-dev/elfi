"""This module implements testbench-functionality in elfi."""

import logging

import numpy as np

from elfi.visualization.visualization import ProgressBar

logger = logging.getLogger(__name__)

__all__ = ['Testbench', 'TestbenchMethod']


class Testbench:
    """Base class for comparing the performance of LFI-methods.

       One elfi.Model can be inferred `repetitions`-times with
       each of the methods included in `method_list`.

    Attributes
    ----------
    model : elfi.Model
        elfi.Model which is inferred.
    method_list : list
        List of elfi-inference methods.
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
            elfi.Model which is inferred.
        repetitions : int
            How many repetitions of models is included in the testbench.
        observation : np.array, optional
            Observation, if available.
        reference_parameter : dictionary, optional
            True parameter values if available.
        reference_posterior : np.array, optional
            A sample from a reference posterior.
        progress_bar : boolean
            Indicate whether to display testbench progressbar.
        seed : int, optional

        """
        # TODO: Resolve the situation when the name of the method to be added already exists.
        self.model = model
        self.method_list = []
        self.method_seed_list = []
        self.repetitions = repetitions
        self.rng = np.random.RandomState(seed)

        if observations is not None:
            self.observations = observations.copy()
        else:
            self.observations = observations

        if reference_parameter is not None:
            self.reference_parameter = reference_parameter.copy()
        else:
            self.reference_parameter = reference_parameter

        self.param_dim = len(model.parameter_names)
        self.param_names = model.parameter_names
        # TODO Add functionality to deal with reference posterior
        self.reference_posterior = reference_posterior
        self.simulator_name = list(model.observed)[0]
        if progress_bar:
            self.progress_bar = ProgressBar(prefix='Progress', suffix='Complete',
                                            decimals=1, length=50, fill='=')
        else:
            self.progress_bar = None

        self._resolve_test_type()
        self._collect_tests()

    def _collect_tests(self):
        self.test_dictionary = {
            'model': self.model,
            'observations': self.observations,
            'reference_parameter': self.reference_parameter,
            'reference_posterior': self.reference_posterior
        }

    def _get_seeds(self, n_rep=1):
        """Fix a seed for each of the repeated instances."""
        upper_limit = 2 ** 32 - 1
        return self.rng.randint(
            low=0,
            high=upper_limit,
            size=n_rep,
            dtype=np.uint32)

    def _resolve_test_type(self):
        self._set_default_test_type()
        self._resolve_observations()
        self._resolve_reference_parameters()

    def _set_default_test_type(self):
        self.description = {
            'observations_available': self.observations is not None,
            'reference_parameters_available': self.reference_parameter is not None,
            'reference_posterior_available': self.reference_posterior is not None
            }

    def _resolve_reference_parameters(self):
        if self.description['reference_parameters_available']:
            for keys, values in self.reference_parameter.items():
                self.reference_parameter[keys] = np.repeat(
                    values,
                    repeats=self.repetitions
                    )

        elif not self.description['observations_available']:
            seed = self._get_seeds(n_rep=1)
            self.reference_parameter = self.model.generate(
                batch_size=self.repetitions,
                outputs=self.model.parameter_names,
                seed=seed[0])

    def _resolve_observations(self):
        if self.description['observations_available']:
            self.observations = np.repeat(
                self.observations,
                repeats=self.repetitions,
                axis=0)
        else:
            seed = self._get_seeds(n_rep=1)
            self.observations = self.model.generate(
                with_values=self.reference_parameter,
                outputs=self.simulator_name,
                batch_size=self.repetitions,
                seed=seed[0])[self.simulator_name]

    def add_method(self, new_method):
        """Add a new method to the testbench.

        Parameters
        ----------
        new_method : TestbenchMethod
            An inference method as a TestbenchMethod.

        """
        logger.info('Adding {} to testbench.'.format(new_method.attributes['name']))
        self.method_list.append(new_method)
        self.method_seed_list.append(self._get_seeds(n_rep=self.repetitions))

    def run(self):
        """Run Testbench."""
        self.testbench_results = []
        for method_index, method in enumerate(self.method_list):
            logger.info('Running {} in testbench.'.format(method.attributes['name']))

            if self.progress_bar:
                self.progress_bar.reinit_progressbar(reinit_msg=method.attributes['name'])

            self.testbench_results.append(
                self._repeat_inference(method, self.method_seed_list[method_index])
                )

    def _repeat_inference(self, method, seed_list):
        repeated_result = []
        model = self.model.copy()
        for i in np.arange(self.repetitions):
            if self.progress_bar:
                self.progress_bar.update_progressbar(i + 1, self.repetitions)

            model.observed[self.simulator_name] = np.atleast_2d(self.observations[i])

            repeated_result.append(
                self._draw_posterior_sample(method, model, seed_list[i])
                )

        return self._collect_results(
            method.attributes['name'],
            repeated_result)

    def _draw_posterior_sample(self, method, model, seed):
        method_instance = method.attributes['callable'](
            model,
            **method.attributes['method_kwargs'],
            seed=seed)

        fit_kwargs = method.attributes['fit_kwargs']

        if len(fit_kwargs) > 0:
            method_instance.fit(fit_kwargs)

        sampler_kwargs = method.attributes['sample_kwargs']

        return method_instance.sample(**sampler_kwargs)

    def _collect_results(self, name, results):
        result_dictionary = {
            'method': name,
            'results': results
        }
        return result_dictionary

    # TODO
    def _compare_sample_results(self):
        """Compare results in sample-format."""

    # TODO
    def _retrodiction(self):
        """Infer a problem with known parameter values."""

    def get_testbench_results(self):
        """Return Testbench testcases and results."""
        testbench_data = {
            'testcases': self.test_dictionary,
            'results': self.testbench_results
        }
        return testbench_data

    def parameterwise_sample_mean_differences(self):
        """Return parameterwise differences for the sample mean for methods in Testbench."""
        sample_mean_difference_results = {}
        for _, method_results in enumerate(self.testbench_results):
            sample_mean_difference_results[method_results['method']] = (
                self._get_sample_mean_difference(method_results)
            )

        return sample_mean_difference_results

    def _get_sample_mean_difference(self, method):
        sample_mean_difference = {}
        for param_names in self.param_names:
            sample_mean_difference[param_names] = [
                results.sample_means[param_names] - self.reference_parameter[param_names][0]
                for results in method['results']
            ]

        return sample_mean_difference


class TestbenchMethod:
    """Container for ParameterInference methods included in Testbench."""

    def __init__(self,
                 method,
                 method_kwargs={},
                 fit_kwargs={},
                 sample_kwargs={},
                 name=None):
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

        """
        name = name or method.__name__
        self.attributes = {'callable': method,
                           'method_kwargs': method_kwargs,
                           'fit_kwargs': fit_kwargs,
                           'sample_kwargs': sample_kwargs,
                           'name': name}

    def set_method_kwargs(self, **kwargs):
        """Add options for the ParameterInference contructor."""
        logger.info("Setting options for {}".format(self.attributes['name']))
        self.attributes['method_kwargs'] = kwargs

    def set_fit_kwargs(self, **kwargs):
        """Add options for the ParameterInference method fit()."""
        logger.info("Setting surrogate fit options for {}".format(self.attributes['name']))
        self.attributes['fit_kwargs'] = kwargs

    def set_sample_kwargs(self, **kwargs):
        """Add options for the ParameterInference method sample()."""
        logger.info("Setting sampler options for {}".format(self.attributes['name']))
        self.attributes['sample_kwargs'] = kwargs

    def get_method(self):
        """Return TestbenchMethod attributes."""
        return self.attributes
