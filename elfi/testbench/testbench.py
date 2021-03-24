"""This module implements testbench-functionality to elfi"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['TestBench']


class TestBench:
    """Base class for comparing the performance of LFI-methods.

    Attributes
    ----------
    model_list : list
        List of elfi.Models that are compared.
    method_list : list
        List of elfi-infernce methods.
    repetitions : int
        How many repetitions of models is included in the testbench.
    seed : int, optional


    """

    def __init__(self,
                 model_list=None,
                 repetitions=1,
                 seed=None):
        """Construct the testbench object.

        Parameters
        ----------
        model_list : list
            List of elfi.Models that are compared.
        repetitions : int
            How many repetitions of models is included in the testbench.
        seed : int, optional

        """

        self.list_of_models = list_of_models
        self.list_of_methods = []
        self.repetitions = repetitions
        self.seed = seed

    def add_model(self, new_model):
        """Add a new model to the testbench.

        Parameters
        ----------
        new_model : elfi.Model
            An elfi.Model object.

        """
        self.list_of_methods.append(new_method)

    def add_method(self, new_method):
        """Add a new method to the testbench.

        Parameters
        ----------
        new_method : ParameterInference
            An inference method subclassed from ParameterInference.

        """
        self.list_of_methods.append(new_method)

    def run_testbench(self):
        for model_index, model in enumerate(self.model_list):
            for method_index, method in enumerate(self.method_list):

                current_method = method['name'](model, **method['method_kwargs'])

                fit_kwargs = method['fit_kwargs']
                sampler_kwargs = method['sampler_kwargs']

                if len(fit_kwargs) > 0:
                    current_method.fit(fit_kwargs)

                method_samples = current_method.sample(1000, **sampler_kwargs)
                print(smc_samples)
    #smc = elfi.SMC(li



    def _compare_sample_results(self):
        """Method for comparing results in sample-format."""


    def _retro_fitting(self):
        """Infer a problem with known parameter values."""


class TestbenchMethod:
    """Container for Inference methods used in TestBench."""
    def __init__(self,
                 name,
                 method_kwargs={},
                 fit_kwargs={},
                 sampler_kwargs={},
                 seed=None):
        method = {'name': name,
                  'method_kwargs': method_kwargs,
                  'fit_kwargs': fit_kwargs,
                  'sampler_kwargs': sampler_kwargs}



class GroundTruth:
    """Base class the ground truth solution."""


class GroundTruthParameter(GroundTruth):


class GroundTruthPSample(GroundTruth):


class GroundTruthObservation:


class GroundTruthPredictedSample(GroundTruthObservation):