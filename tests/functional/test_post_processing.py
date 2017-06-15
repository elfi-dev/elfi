import numpy as np

import elfi
from elfi.examples import bdm, ma2
from elfi import LinearAdjustment
from elfi.methods.post_processing import adjust_posterior


def _statistics(arr):
    return arr.mean(), arr.var()


def test_single_parameter_linear_adjustment():
    """A regression test against values obtained in the notebook."""
    m = bdm.get_model(alpha=0.2, delta=0, tau=0.198, N=20, seed_obs=None)
    seed = 20170511
    threshold = 0.2
    batch_size = 1000
    n_samples = 500

    summary_names = ['T1']
    parameter_names = ['alpha']
    linear_adjustment = LinearAdjustment()

    res = elfi.Rejection(m['d'], batch_size=batch_size,
                         outputs=['T1'],
                         # outputs=summary_names,
                         seed=seed).sample(n_samples, threshold=threshold)
    adjusted = adjust_posterior(model=m, result=res,
                                parameter_names=parameter_names,
                                summary_names=summary_names,
                                adjustment=linear_adjustment)
    alpha = adjusted.outputs['alpha']

    ref_mean, ref_var = (0.35199354166477109, 0.034264439593055904)
    assert _statistics(alpha) == (ref_mean, ref_var)


def test_multi_parameter_linear_adjustment():
    """A regression test against values obtained in the notebook."""
    seed = 20170511
    threshold = 0.2
    batch_size = 1000
    n_samples = 500
    m = ma2.get_model(true_params=[0.6, 0.2], seed_obs=seed)

    summary_names = ['S1', 'S2']
    parameter_names = ['t1', 't2']
    linear_adjustment = LinearAdjustment()

    res = elfi.Rejection(m['d'], batch_size=batch_size,
                         outputs=['S1', 'S2'],
                         # outputs=summary_names, # fails ?!?!?
                         seed=seed).sample(n_samples, threshold=threshold)
    adjusted = adjust_posterior(model=m, result=res,
                                parameter_names=parameter_names,
                                summary_names=summary_names,
                                adjustment=linear_adjustment)
    t1 = adjusted.outputs['t1']
    t2 = adjusted.outputs['t2']

    t1_mean, t1_var = (0.51606048286584782, 0.017253007645871756)
    t2_mean, t2_var = (0.15805189695581101, 0.028004406914362647)
    assert _statistics(t1) == (t1_mean, t1_var)
    assert _statistics(t2) == (t2_mean, t2_var)
