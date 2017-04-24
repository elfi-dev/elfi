import pytest
import logging

import numpy as np

import elfi

from elfi.methods.methods import InferenceMethod


def test_no_model_parameters(simple_model):
    simple_model.parameters = None

    with pytest.raises(Exception):
        InferenceMethod(simple_model, [])


@pytest.mark.usefixtures('with_all_clients')
def test_smc_prior_use(ma2):
    thresholds = [.5]
    N = 1000
    smc = elfi.SMC(ma2['d'], batch_size=20000)
    res = smc.sample(N, thresholds=thresholds)
    dens = res.populations[0].outputs['_prior_pdf']
    # Test that the density is uniform
    assert np.allclose(dens, dens[0])

