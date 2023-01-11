This tutorial is generated from a `Jupyter <http://jupyter.org/>`__
notebook that can be found
`here <https://github.com/elfi-dev/notebooks>`__.

BOLFI
-----

In practice inference problems often have a complicated and
computationally heavy simulator, and one simply cannot run it for
millions of times. The Bayesian Optimization for Likelihood-Free
Inference `BOLFI <http://jmlr.csail.mit.edu/papers/v17/15-017.html>`__
framework is likely to prove useful in such situation: a statistical
model (usually `Gaussian
process <https://en.wikipedia.org/wiki/Gaussian_process>`__, GP) is
created for the discrepancy, and its minimum is inferred with `Bayesian
optimization <https://en.wikipedia.org/wiki/Bayesian_optimization>`__.
This approach typically reduces the number of required simulator calls
by several orders of magnitude.

This tutorial demonstrates how to use BOLFI to do LFI in ELFI.

.. code:: ipython3

    import numpy as np
    import scipy.stats
    import matplotlib
    import matplotlib.pyplot as plt
    
    %matplotlib inline
    %precision 2
    
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Set an arbitrary global seed to keep the randomly generated quantities the same
    seed = 1
    np.random.seed(seed)
    
    import elfi

Although BOLFI is best used with complicated simulators, for
demonstration purposes we will use the familiar MA2 model introduced in
the basic tutorial, and load it from ready-made examples:

.. code:: ipython3

    from elfi.examples import ma2
    model = ma2.get_model(seed_obs=seed)
    elfi.draw(model)




.. image:: https://raw.githubusercontent.com/elfi-dev/notebooks/dev/figures/BOLFI_files/BOLFI_5_0.svg



Fitting the surrogate model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can immediately proceed with the inference. However, when dealing
with a Gaussian process, it may be beneficial to take a logarithm of the
discrepancies in order to reduce the effect that high discrepancies have
on the GP. (Sometimes you may want to add a small constant to avoid very
negative or even -Inf distances occurring especially if it is likely
that there can be exact matches between simulated and observed data.) In
ELFI such transformed node can be created easily:

.. code:: ipython3

    log_d = elfi.Operation(np.log, model['d'])

As BOLFI is a more advanced inference method, its interface is also a
bit more involved as compared to for example rejection sampling. But not
much: Using the same graphical model as earlier, the inference could
begin by defining a Gaussian process (GP) model, for which ELFI uses the
`GPy <https://sheffieldml.github.io/GPy/>`__ library. This could be
given as an ``elfi.GPyRegression`` object via the keyword argument
``target_model``. In this case, we are happy with the default that ELFI
creates for us when we just give it each parameter some ``bounds`` as a
dictionary.

Other notable arguments include the ``initial_evidence``, which gives
the number of initialization points sampled straight from the priors
before starting to optimize the acquisition of points,
``update_interval`` which defines how often the GP hyperparameters are
optimized, and ``acq_noise_var`` which defines the diagonal covariance
of noise added to the acquired points. Note that in general BOLFI does
not benefit from a ``batch_size`` higher than one, since the acquisition
surface is updated after each batch (especially so if the noise is 0!).

.. code:: ipython3

    bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20, update_interval=10, 
                       bounds={'t1':(-2, 2), 't2':(-1, 1)}, acq_noise_var=[0.1, 0.1], seed=seed)

Sometimes you may have some samples readily available. You could then
initialize the GP model with a dictionary of previous results by giving
``initial_evidence=result.outputs``.

The BOLFI class can now try to ``fit`` the surrogate model (the GP) to
the relationship between parameter values and the resulting
discrepancies. We’ll request only 100 evidence points (including the
``initial_evidence`` defined above).

.. code:: ipython3

    %time post = bolfi.fit(n_evidence=200)


.. parsed-literal::

    INFO:elfi.methods.parameter_inference:BOLFI: Fitting the surrogate model...
    INFO:elfi.methods.posteriors:Using optimized minimum value (-1.6146) of the GP discrepancy mean function as a threshold


.. parsed-literal::

    CPU times: user 1min 48s, sys: 1.29 s, total: 1min 50s
    Wall time: 1min


(More on the returned ``BolfiPosterior`` object
`below <#BOLFI-Posterior>`__.)

Note that in spite of the very few simulator runs, fitting the model
took longer than any of the previous methods. Indeed, BOLFI is intended
for scenarios where the simulator takes a lot of time to run.

The fitted ``target_model`` uses the GPy library, and can be
investigated further:

.. code:: ipython3

    bolfi.target_model




.. parsed-literal::

    
    Name : GP regression
    Objective : 151.86636065302943
    Number of Parameters : 4
    Number of Optimization Parameters : 4
    Updates : True
    Parameters:
      [1mGP_regression.         [0;0m  |           value  |  constraints  |     priors   
      [1msum.rbf.variance       [0;0m  |  0.321697451372  |      +ve      |  Ga(0.024, 1)
      [1msum.rbf.lengthscale    [0;0m  |  0.541352150083  |      +ve      |   Ga(1.3, 1) 
      [1msum.bias.variance      [0;0m  |  0.021827430988  |      +ve      |  Ga(0.006, 1)
      [1mGaussian_noise.variance[0;0m  |  0.183562040169  |      +ve      |              



.. code:: ipython3

    bolfi.plot_state();



.. parsed-literal::

    <matplotlib.figure.Figure at 0x11b2b2ba8>



.. image:: https://raw.githubusercontent.com/elfi-dev/notebooks/dev/figures/BOLFI_files/BOLFI_15_1.png


It may be useful to see the acquired parameter values and the resulting
discrepancies:

.. code:: ipython3

    bolfi.plot_discrepancy();



.. image:: https://raw.githubusercontent.com/elfi-dev/notebooks/dev/figures/BOLFI_files/BOLFI_17_0.png


There could be an unnecessarily high number of points at parameter
bounds. These could probably be decreased by lowering the covariance of
the noise added to acquired points, defined by the optional
``acq_noise_var`` argument for the BOLFI constructor. Another
possibility could be to `add virtual derivative observations at the
borders <https://arxiv.org/abs/1704.00963>`__, though not yet
implemented in ELFI.

BOLFI Posterior
~~~~~~~~~~~~~~~

Above, the ``fit`` method returned a ``BolfiPosterior`` object
representing a BOLFI posterior (please see the
`paper <http://jmlr.csail.mit.edu/papers/v17/15-017.html>`__ for
details). The ``fit`` method accepts a threshold parameter; if none is
given, ELFI will use the minimum value of discrepancy estimate mean.
Afterwards, one may request for a posterior with a different threshold:

.. code:: ipython3

    post2 = bolfi.extract_posterior(-1.)

One can visualize a posterior directly (remember that the priors form a
triangle):

.. code:: ipython3

    post.plot(logpdf=True)



.. image:: https://raw.githubusercontent.com/elfi-dev/notebooks/dev/figures/BOLFI_files/BOLFI_23_0.png


Sampling
~~~~~~~~

Finally, samples from the posterior can be acquired with an MCMC
sampler. By default it runs 4 chains, and half of the requested samples
are spent in adaptation/warmup. Note that depending on the smoothness of
the GP approximation, the number of priors, their gradients etc., **this
may be slow**.

.. code:: ipython3

    %time result_BOLFI = bolfi.sample(1000, info_freq=1000)


.. parsed-literal::

    INFO:elfi.methods.posteriors:Using optimized minimum value (-1.6146) of the GP discrepancy mean function as a threshold
    INFO:elfi.methods.mcmc:NUTS: Performing 1000 iterations with 500 adaptation steps.
    INFO:elfi.methods.mcmc:NUTS: Adaptation/warmup finished. Sampling...
    INFO:elfi.methods.mcmc:NUTS: Acceptance ratio: 0.423. After warmup 68 proposals were outside of the region allowed by priors and rejected, decreasing acceptance ratio.
    INFO:elfi.methods.mcmc:NUTS: Performing 1000 iterations with 500 adaptation steps.
    INFO:elfi.methods.mcmc:NUTS: Adaptation/warmup finished. Sampling...
    INFO:elfi.methods.mcmc:NUTS: Acceptance ratio: 0.422. After warmup 71 proposals were outside of the region allowed by priors and rejected, decreasing acceptance ratio.
    INFO:elfi.methods.mcmc:NUTS: Performing 1000 iterations with 500 adaptation steps.
    INFO:elfi.methods.mcmc:NUTS: Adaptation/warmup finished. Sampling...
    INFO:elfi.methods.mcmc:NUTS: Acceptance ratio: 0.419. After warmup 65 proposals were outside of the region allowed by priors and rejected, decreasing acceptance ratio.
    INFO:elfi.methods.mcmc:NUTS: Performing 1000 iterations with 500 adaptation steps.
    INFO:elfi.methods.mcmc:NUTS: Adaptation/warmup finished. Sampling...
    INFO:elfi.methods.mcmc:NUTS: Acceptance ratio: 0.439. After warmup 66 proposals were outside of the region allowed by priors and rejected, decreasing acceptance ratio.


.. parsed-literal::

    4 chains of 1000 iterations acquired. Effective sample size and Rhat for each parameter:
    t1 2222.1197791 1.00106816947
    t2 2256.93599184 1.0003364409
    CPU times: user 1min 45s, sys: 1.29 s, total: 1min 47s
    Wall time: 55.1 s


The sampling algorithms may be fine-tuned with some parameters. The
default
`No-U-Turn-Sampler <http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf>`__
is a sophisticated algorithm, and in some cases one may get warnings
about diverged proposals, which are signs that `something may be wrong
and should be
investigated <http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup>`__.
It is good to understand the cause of these warnings although they don’t
automatically mean that the results are unreliable. You could try
rerunning the ``sample`` method with a higher target probability
``target_prob`` during adaptation, as its default 0.6 may be inadequate
for a non-smooth posteriors, but this will slow down the sampling.

Note also that since MCMC proposals outside the region allowed by either
the model priors or GP bounds are rejected, a tight domain may lead to
suboptimal overall acceptance ratio. In our MA2 case the prior defines a
triangle-shaped uniform support for the posterior, making it a good
example of a difficult model for the NUTS algorithm.

Now we finally have a ``Sample`` object again, which has several
convenience methods:

.. code:: ipython3

    result_BOLFI




.. parsed-literal::

    Method: BOLFI
    Number of samples: 2000
    Number of simulations: 200
    Threshold: -1.61
    Sample means: t1: 0.429, t2: 0.0277



.. code:: ipython3

    result_BOLFI.plot_traces();



.. image:: https://raw.githubusercontent.com/elfi-dev/notebooks/dev/figures/BOLFI_files/BOLFI_29_0.png


The black vertical lines indicate the end of warmup, which by default is
half of the number of iterations.

.. code:: ipython3

    result_BOLFI.plot_marginals();



.. image:: https://raw.githubusercontent.com/elfi-dev/notebooks/dev/figures/BOLFI_files/BOLFI_31_0.png

