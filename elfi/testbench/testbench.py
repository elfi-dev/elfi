"""This module implements testbench-functionality to elfi"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['TestBench']


class TestBench:
    """Base class for comparing the performance of LFI-methods.

    Attributes
    ----------
    list_of_models : list
        List of elfi.Models that are compared.
    list_of_methods : list
        List of elfi-infernce methods.
    repetitions : int
        How many repetitions of models is included in the testbench.
    seed : int, optional


    """

    def __init__(self,
                 list_of_models,
                 list_of_methods,
                 repetitions=1,
                 seed=None):
        """Construct the testbench object.

        Parameters
        ----------
        list_of_models : list
            List of elfi.Models that are compared.
        list_of_methods : list
            List of elfi-infernce methods.
        repetitions : int
            How many repetitions of models is included in the testbench.
        seed : int, optional


        """

    def _compare_sample_results(self):
        """Method for comparing results in sample-format."""


    def _retro_fitting(self):
        """Infer a problem with known parameter values."""


class GroundTruth:
    """Base class the ground truth solution."""


class GroundTruthParameter(GroundTruth):


class GroundTruthPSample(GroundTruth):


class GroundTruthObservation:


class GroundTruthPredictedSample(GroundTruthObservation):