import numpy as np
from numpy.lib.function_base import quantile
import pytest

import elfi
import elfi.examples.ma2 as exma2
from elfi.methods.inference.parameter_inference import ParameterInference


def test_testbenchmethod_init():

    method = elfi.TestbenchMethod(method=elfi.SMC, name="SMC_1")
    method.set_method_kwargs(discrepancy_name='d', batch_size=50)
    method.set_sample_kwargs(n_samples=100, thresholds=[2.0, 1.0], bar=False)
    attr = method.get_method()

    assert attr['name'] == "SMC_1"
    assert attr['method_kwargs']['batch_size'] == 50
    assert attr['sample_kwargs']['n_samples'] == 100

def test_testbench_init_param_reps(ma2):

    testbench = elfi.Testbench(model=ma2,
                               repetitions=5,
                               seed=99,
                               progress_bar=False)

    for _, values in testbench.reference_parameter.items():
        assert values.size == 5


def test_testbench_init_given_params(ma2):

    ref_params = ma2.generate(batch_size=1, outputs=['t1', 't2'])
    testbench = elfi.Testbench(model=ma2,
                               reference_parameter=ref_params,
                               repetitions=5,
                               seed=99,
                               progress_bar=False)

    for _, values in testbench.reference_parameter.items():
        assert np.all(values == values[0])
        assert values.size == 5


def test_testbench_init_obs_reps(ma2):

    testbench = elfi.Testbench(model=ma2,
                               repetitions=5,
                               seed=99,
                               progress_bar=False)

    assert len(testbench.observations) == 5


def test_testbench_init_given_obs(ma2):

    obs = ma2.generate(batch_size=1, outputs=['MA2'])
    testbench = elfi.Testbench(model=ma2,
                               observations=obs,
                               repetitions=5,
                               seed=99,
                               progress_bar=False)

    assert len(testbench.observations) == 5
    assert np.all(
        [a == b for a, b in zip([obs], testbench.observations)]
        )


def test_testbench_execution(ma2):

    method1 = elfi.TestbenchMethod(method=elfi.Rejection, name='Rejection_1')
    method1.set_method_kwargs(discrepancy_name='d', batch_size=500)
    method1.set_sample_kwargs(n_samples=500, bar=False)

    method2 = elfi.TestbenchMethod(method=elfi.Rejection, name='Rejection_2')
    method2.set_method_kwargs(discrepancy_name='d', batch_size=500)
    method2.set_sample_kwargs(n_samples=500, quantile=0.5, bar=False)

    testbench = elfi.Testbench(model=ma2,
                               repetitions=3,
                               seed=156,
                               progress_bar=False)
    testbench.add_method(method1)
    testbench.add_method(method2)

    testbench.run()

    sample_mean_differences = testbench.parameterwise_sample_mean_differences()
    assert len(sample_mean_differences) == 2
    assert len(sample_mean_differences['Rejection_1']) == 2
    assert len(sample_mean_differences['Rejection_1']['t1']) == 3


def test_testbench_seeding(ma2):

    testbench1 = elfi.Testbench(model=ma2,
                                repetitions=2,
                                seed=100,
                                progress_bar=False)

    testbench2 = elfi.Testbench(model=ma2,
                                repetitions=2,
                                seed=100,
                                progress_bar=False)

    assert len(testbench1.observations) == len(testbench2.observations)
    assert np.all(
        [a == b for a, b in zip(testbench1.observations, testbench2.observations)]
        )
