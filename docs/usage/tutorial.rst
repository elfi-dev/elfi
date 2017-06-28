
This tutorial is generated from a `Jupyter <http://jupyter.org/>`__
notebook that can be found
`here <https://github.com/elfi-dev/notebooks>`__.

ELFI tutorial
=============

This tutorial covers the basics of using ELFI, i.e. how to make models,
save results for later use and run different inference algorithms.

Let's begin by importing libraries that we will use and specify some
settings.

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
    np.random.seed(20170530)

Inference with ELFI: case MA(2) model
-------------------------------------

Throughout this tutorial we will use the 2nd order moving average model
MA(2) as an example. MA(2) is a common model used in univariate time
series analysis. Assuming zero mean it can be written as

.. math::


   y_t = w_t + \theta_1 w_{t-1} + \theta_2 w_{t-2},

where :math:`\theta_1, \theta_2 \in \mathbb{R}` and
:math:`(w_k)_{k\in \mathbb{Z}} \sim N(0,1)` represents an independent
and identically distributed sequence of white noise.

The observed data and the inference problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial, our task is to infer the parameters
:math:`\theta_1, \theta_2` given a sequence of 100 observations
:math:`y` that originate from an MA(2) process. Let's define the MA(2)
simulator as a Python function:

.. code:: ipython3

    def MA2(t1, t2, n_obs=100, batch_size=1, random_state=None):
        # Make inputs 2d arrays for numpy broadcasting with w
        t1 = np.asanyarray(t1).reshape((-1, 1))
        t2 = np.asanyarray(t2).reshape((-1, 1))
        random_state = random_state or np.random
    
        w = random_state.randn(batch_size, n_obs+2)  # i.i.d. sequence ~ N(0,1)
        x = w[:, 2:] + t1*w[:, 1:-1] + t2*w[:, :-2]
        return x

Above, ``t1``, ``t2``, and ``n_obs`` are the arguments specific to the
MA2 process. The latter two, ``batch_size`` and ``random_state`` are
ELFI specific keyword arguments. The ``batch_size`` argument tells how
many simulations are needed. The ``random_state`` argument is for
generating random quantities in your simulator. It is a
``numpy.RandomState`` object that has all the same methods as
``numpy.random`` module has. It is used for ensuring consistent results
and handling random number generation in parallel settings.

Vectorization
~~~~~~~~~~~~~

What is the purpose of the ``batch_size`` argument? In ELFI, operations
are vectorized, meaning that instead of simulating a single MA2 sequence
at a time, we simulate a batch of them. A vectorized function takes
vectors as inputs, and computes the output for each element in the
vector. Vectorization is a way to make operations efficient in Python.
Above we rely on numpy to carry out the vectorized calculations.

In this case the arguments ``t1`` and ``t2`` are going to be vectors of
length ``batch_size`` and the method returns a 2d array with the
simulations on the rows. Notice that for convenience, the funtion also
works with scalars that are first converted to vectors.

.. note:: there is a built-in tool (`elfi.tools.vectorize`) in ELFI to vectorize operations that are not vectorized. It is basically a for loop wrapper.

.. Important:: in order to guarantee a consistent state of pseudo-random number generation, the simulator must have `random_state` as a keyword argument for reading in a `numpy.RandomState` object.

Let's now use this simulator to create toy observations. We will use
parameter values :math:`\theta_1=0.6, \theta_2=0.2` as in `*Marin et al.
(2012)* <http://link.springer.com/article/10.1007/s11222-011-9288-2>`__
and then try to infer these parameter values back based on the toy
observed data alone.

.. code:: ipython3

    # true parameters
    t1_true = 0.6
    t2_true = 0.2
    
    y_obs = MA2(t1_true, t2_true)
    
    # Plot the observed sequence
    plt.figure(figsize=(11, 6));
    plt.plot(y_obs.ravel());
    
    # To illustrate the stochasticity, let's plot a couple of more observations with the same true parameters:
    plt.plot(MA2(t1_true, t2_true).ravel());
    plt.plot(MA2(t1_true, t2_true).ravel());



.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_10_0.png


Approximate Bayesian Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard statistical inference methods rely on the use of the
*likelihood* function. Given a configuration of the parameters, the
likelihood function quantifies how likely it is that values of the
parameters produced the observed data. In our simple example case above
however, evaluating the likelihood is difficult due to the unobserved
latent sequence (variable ``w`` in the simulator code). In many real
world applications the likelihood function is not available or it is too
expensive to evaluate preventing the use of traditional inference
methods.

One way to approach this problem is to use Approximate Bayesian
Computation (ABC) which is a statistically based method replacing the
use of the likelihood function with a simulator of the data. Loosely
speaking, it is based on the intuition that similar data is likely to
have been produced by similar parameters. Looking at the picture above,
in essence we would keep simulating until we have found enough sequences
that are similar to the observed sequence. Although the idea may appear
inapplicable for the task at hand, you will soon see that it does work.
For more information about ABC, please see e.g.

-  `Lintusaari, J., Gutmann, M. U., Dutta, R., Kaski, S., and Corander,
   J. (2016). Fundamentals and recent developments in approximate
   Bayesian computation. *Systematic Biology*, doi:
   10.1093/sysbio/syw077. <http://sysbio.oxfordjournals.org/content/early/2016/09/07/sysbio.syw077.full.pdf>`__

-  `Marin, J.-M., Pudlo, P., Robert, C. P., and Ryder, R. J. (2012).
   Approximate Bayesian computational methods. *Statistics and
   Computing*,
   22(6):1167–1180. <http://link.springer.com/article/10.1007/s11222-011-9288-2>`__

-  https://en.wikipedia.org/wiki/Approximate\_Bayesian\_computation

Defining the model
------------------

ELFI includes an easy to use generative modeling syntax, where the
generative model is specified as a directed acyclic graph
(`DAG <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`__). This
provides an intuitive means to describe rather complex dependencies
conveniently. Often the target of the generative model is a distance
between the simulated and observed data. To start creating our model, we
will first import ELFI:

.. code:: ipython3

    import elfi

As is usual in Bayesian statistical inference, we need to define *prior*
distributions for the unknown parameters :math:`\theta_1, \theta_2`. In
ELFI the priors can be any of the continuous and discrete distributions
available in ``scipy.stats`` (for custom priors, see
`below <#Custom-priors>`__). For simplicity, let's start by assuming
that both parameters follow ``Uniform(0, 2)``.

.. code:: ipython3

    # a node is defined by giving a distribution from scipy.stats together with any arguments (here 0 and 2)
    t1 = elfi.Prior(scipy.stats.uniform, 0, 2)
    
    # ELFI also supports giving the scipy.stats distributions as strings
    t2 = elfi.Prior('uniform', 0, 2)

Next, we define the *simulator* node with the ``MA2`` function above,
and give the priors to it as arguments. This means that the parameters
for the simulations will be drawn from the priors. Because we have the
observed data available for this node, we provide it here as well:

.. code:: ipython3

    Y = elfi.Simulator(MA2, t1, t2, observed=y_obs)

But how does one compare the simulated sequences with the observed
sequence? Looking at the plot of just a few observed sequences above, a
direct pointwise comparison would probably not work very well: the three
sequences look quite different although they were generated with the
same parameter values. Indeed, the comparison of simulated sequences is
often the most difficult (and ad hoc) part of ABC. Typically one chooses
one or more summary statistics and then calculates the discrepancy
between those.

Here, we will apply the intuition arising from the definition of the
MA(2) process, and use the autocovariances with lags 1 and 2 as the
summary statistics:

.. code:: ipython3

    def autocov(x, lag=1):
        C = np.mean(x[:,lag:] * x[:,:-lag], axis=1)
        return C

As is familiar by now, a ``Summary`` node is defined by giving the
autocovariance function and the simulated data (which includes the
observed as well):

.. code:: ipython3

    S1 = elfi.Summary(autocov, Y)
    S2 = elfi.Summary(autocov, Y, 2)  # the optional keyword lag is given the value 2

Here, we choose the discrepancy as the common Euclidean L2-distance.
ELFI can use many common distances directly from
``scipy.spatial.distance`` like this:

.. code:: ipython3

    # Finish the model with the final node that calculates the squared distance (S1_sim-S1_obs)**2 + (S2_sim-S2_obs)**2
    d = elfi.Distance('euclidean', S1, S2)

One may wish to use a distance function that is unavailable in
``scipy.spatial.distance``. ELFI supports defining a custom
distance/discrepancy functions as well (see the documentation for
``elfi.Distance`` and ``elfi.Discrepancy``).

Now that the inference model is defined, ELFI can visualize the model as
a DAG.

.. code:: ipython3

    elfi.draw(d)  # just give it a node in the model, or the model itself (d.model)




.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_27_0.svg



.. note:: You will need the Graphviz_ software as well as the graphviz `Python package`_ (https://pypi.python.org/pypi/graphviz) for drawing this. The software is already installed in many unix-like OS.

.. _Graphviz: http://www.graphviz.org
.. _`Python package`: https://pypi.python.org/pypi/graphviz

Modifying the model
-------------------

Although the above definition is perfectly valid, let's use the same
priors as in `*Marin et al.
(2012)* <http://link.springer.com/article/10.1007/s11222-011-9288-2>`__
that guarantee that the problem will be identifiable (loosely speaking,
the likelihood willl have just one mode). Marin et al. used priors for
which :math:`-2<\theta_1<2` with :math:`\theta_1+\theta_2>-1` and
:math:`\theta_1-\theta_2<1` i.e. the parameters are sampled from a
triangle (see below).

Custom priors
~~~~~~~~~~~~~

In ELFI, custom distributions can be defined similar to distributions in
``scipy.stats`` (i.e. they need to have at least the ``rvs`` method
implemented for the simplest algorithms). To be safe they can inherit
``elfi.Distribution`` which defines the methods needed. In this case we
only need these for sampling, so implementing a static ``rvs`` method
suffices. As was in the context of simulators, it is important to accept
the keyword argument ``random_state``, which is needed for ELFI's
internal book-keeping of pseudo-random number generation. Also the
``size`` keyword is needed (which in the simple cases is the same as the
``batch_size`` in the simulator definition).

.. code:: ipython3

    # define prior for t1 as in Marin et al., 2012 with t1 in range [-b, b]
    class CustomPrior_t1(elfi.Distribution):
        def rvs(b, size=1, random_state=None):
            u = scipy.stats.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
            t1 = np.where(u<0.5, np.sqrt(2.*u)*b-b, -np.sqrt(2.*(1.-u))*b+b)
            return t1
    
    # define prior for t2 conditionally on t1 as in Marin et al., 2012, in range [-a, a]
    class CustomPrior_t2(elfi.Distribution):
        def rvs(t1, a, size=1, random_state=None):
            locs = np.maximum(-a-t1, t1-a)
            scales = a - locs
            t2 = scipy.stats.uniform.rvs(loc=locs, scale=scales, size=size, random_state=random_state)
            return t2

These indeed sample from a triangle:

.. code:: ipython3

    t1_1000 = CustomPrior_t1.rvs(2, 1000)
    t2_1000 = CustomPrior_t2.rvs(t1_1000, 1, 1000)
    plt.scatter(t1_1000, t2_1000, s=4, edgecolor='none');
    # plt.plot([0, 2, -2, 0], [-1, 1, 1, -1], 'b')  # outlines of the triangle



.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_33_0.png


Let's change the earlier priors to the new ones in the inference model:

.. code:: ipython3

    t1.become(elfi.Prior(CustomPrior_t1, 2))
    t2.become(elfi.Prior(CustomPrior_t2, t1, 1))
    
    elfi.draw(d)




.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_35_0.svg



Note that ``t2`` now depends on ``t1``. Yes, ELFI supports hierarchy.

Inference with rejection sampling
---------------------------------

The simplest ABC algorithm samples parameters from their prior
distributions, runs the simulator with these and compares them to the
observations. The samples are either accepted or rejected depending on
how large the distance is. The accepted samples represent samples from
the approximate posterior distribution.

In ELFI, ABC methods are initialized either with a node giving the
distance, or with the ``ElfiModel`` object and the name of the distance
node. Depending on the inference method, additional arguments may be
accepted or required.

A common optional keyword argument, accepted by all inference methods,
``batch_size`` defines how many simulations are performed in each
passing through the graph.

Another optional keyword is the seed. This ensures that the outcome will
be always the same for the same data and model. If you leave it out, a
random seed will be taken.

.. code:: ipython3

    seed = 20170530
    rej = elfi.Rejection(d, batch_size=10000, seed=seed)

.. note:: In Python, doing many calculations with a single function call can potentially save a lot of CPU time, depending on the operation. For example, here we draw 10000 samples from `t1`, pass them as input to `t2`, draw 10000 samples from `t2`, and then use these both to run 10000 simulations and so forth. All this is done in one passing through the graph and hence the overall number of function calls is reduced 10000-fold. However, this does not mean that batches should be as big as possible, since you may run out of memory, the fraction of time spent in function call overhead becomes insignificant, and many algorithms operate in multiples of `batch_size`. Furthermore, the `batch_size` is a crucial element for efficient parallelization (see the notebook on parallelization).

After the ABC method has been initialized, samples can be drawn from it.
By default, rejection sampling in ELFI works in ``quantile`` mode i.e. a
certain quantile of the samples with smallest discrepancies is accepted.
The ``sample`` method requires the number of output samples as a
parameter. Note that the simulator is then run ``(N/quantile)`` times.
(Alternatively, the same behavior can be achieved by saying
``n_sim=1000000``.)

The IPython magic command ``%time`` is used here to give you an idea of
runtime on a typical personal computer. We will turn interactive
visualization on so that if you run this on a notebook you will see the
posterior forming from a prior distribution. In this case most of the
time is spent in drawing.

.. code:: ipython3

    N = 1000
    
    vis = dict(xlim=[-2,2], ylim=[-1,1])
    
    # You can give the sample method a `vis` keyword to see an animation how the prior transforms towards the
    # posterior with a decreasing threshold (interactive visualization will slow it down a bit though).
    %time result = rej.sample(N, quantile=0.01, vis=vis)



.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_42_0.png



.. raw:: html

    <span>Threshold: 0.116859716394976</span>


.. parsed-literal::

    CPU times: user 2.2 s, sys: 182 ms, total: 2.38 s
    Wall time: 2.39 s


The ``sample`` method returns a ``Sample`` object, which contains
several attributes and methods. Most notably the attribute ``samples``
contains an ``OrderedDict`` (i.e. an ordered Python dictionary) of the
posterior numpy arrays for all the model parameters (``elfi.Prior``\ s
in the model). For rejection sampling, other attributes include e.g. the
``threshold``, which is the threshold value resulting in the requested
quantile.

.. code:: ipython3

    result.samples['t1'].mean()




.. parsed-literal::

    0.56



The ``Sample`` object includes a convenient ``summary`` method:

.. code:: ipython3

    result.summary


.. parsed-literal::

    Method: Rejection
    Number of posterior samples: 1000
    Number of simulations: 100000
    Threshold: 0.117
    Posterior means: t1: 0.556, t2: 0.219


Rejection sampling can also be performed with using a threshold or total
number of simulations. Let's define here threshold. This means that all
draws from the prior for which the generated distance is below the
threshold will be accepted as samples. Note that the simulator will run
as long as it takes to generate the requested number of samples.

.. code:: ipython3

    %time result2 = rej.sample(N, threshold=0.2)
    
    print(result2)  # the Sample object's __str__ contains the output from summary()


.. parsed-literal::

    CPU times: user 215 ms, sys: 51.8 ms, total: 267 ms
    Wall time: 269 ms
    Method: Rejection
    Number of posterior samples: 1000
    Number of simulations: 40000
    Threshold: 0.185
    Posterior means: t1: 0.555, t2: 0.223
    


Storing simulated data
----------------------

As the samples are already in numpy arrays, you can just say e.g.
``np.save('t1_data.npy', result.samples['t1'])`` to save them. However,
ELFI provides some additional functionality. You may define a *pool* for
storing all outputs of any node in the model (not just the accepted
samples). Let's save all outputs for ``t1``, ``t2``, ``S1`` and ``S2``
in our model:

.. code:: ipython3

    pool = elfi.OutputPool(['t1', 't2', 'S1', 'S2'])
    rej = elfi.Rejection(d, pool=pool)
    
    %time result3 = rej.sample(N, n_sim=1000000)
    result3


.. parsed-literal::

    CPU times: user 6.14 s, sys: 102 ms, total: 6.24 s
    Wall time: 6.38 s




.. parsed-literal::

    Method: Rejection
    Number of posterior samples: 1000
    Number of simulations: 1000000
    Threshold: 0.036
    Posterior means: t1: 0.56, t2: 0.227



The benefit of the pool is that you may reuse simulations without having
to resimulate them. Above we saved the summaries to the pool, so we can
change the distance node of the model without having to resimulate
anything. Let's do that.

.. code:: ipython3

    # Replace the current distance with a cityblock (manhattan) distance and recreate the inference
    d.become(elfi.Distance('cityblock', S1, S2, p=1))
    rej = elfi.Rejection(d, pool=pool)
    
    %time result4 = rej.sample(N, n_sim=1000000)
    result4


.. parsed-literal::

    CPU times: user 848 ms, sys: 12.1 ms, total: 860 ms
    Wall time: 895 ms




.. parsed-literal::

    Method: Rejection
    Number of posterior samples: 1000
    Number of simulations: 1000000
    Threshold: 0.0447
    Posterior means: t1: 0.56, t2: 0.227



Note the significant saving in time, even though the total number of
considered simulations stayed the same.

We can also continue the inference by increasing the total number of
simulations and only have to simulate the new ones:

.. code:: ipython3

    %time result5 = rej.sample(N, n_sim=1200000)
    result5


.. parsed-literal::

    CPU times: user 1.96 s, sys: 29.4 ms, total: 1.99 s
    Wall time: 2.02 s




.. parsed-literal::

    Method: Rejection
    Number of posterior samples: 1000
    Number of simulations: 1200000
    Threshold: 0.0409
    Posterior means: t1: 0.56, t2: 0.23



Above the results were saved into a python dictionary. If you store a
lot of data to dictionaries, you will eventually run out of memory.
Instead you can save the outputs to standard numpy .npy files:

.. code:: ipython3

    arraypool = elfi.store.ArrayPool(['t1', 't2', 'Y', 'd'], basepath='./output')
    rej = elfi.Rejection(d, pool=arraypool)
    %time result5 = rej.sample(100, threshold=0.3)


.. parsed-literal::

    CPU times: user 25.6 ms, sys: 2.58 ms, total: 28.2 ms
    Wall time: 29.3 ms


This stores the simulated data in binary ``npy`` format under
``arraypool.path``, and can be loaded with ``np.load``.

.. code:: ipython3

    # Let's flush the outputs to disk (alternatively you can close the pool) so that we can read them
    # while we still have the arraypool open.
    arraypool.flush()
    
    import os
    os.listdir(arraypool.path)




.. parsed-literal::

    ['d.npy', 't1.npy', 't2.npy', 'Y.npy']



Now lets load all the parameters ``t1`` that were generated with numpy:

.. code:: ipython3

    np.load(arraypool.path + '/t1.npy')




.. parsed-literal::

    array([-0.51,  0.09,  0.72, ..., -1.23,  0.02, -0.66])



You can delete the files with:

.. code:: ipython3

    arraypool.delete()
    
    # verify the deletion
    try:
        os.listdir(arraypool.path)
        
    except FileNotFoundError:
        print("No such file or directory")


.. parsed-literal::

    No such file or directory


Visualizing the results
-----------------------

Instances of ``Sample`` contain methods for some basic plotting (these
are convenience methods to plotting functions defined under
``elfi.visualization``).

For example one can plot the marginal distributions:

.. code:: ipython3

    result.plot_marginals();



.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_66_0.png


Often "pairwise relationships" are more informative:

.. code:: ipython3

    result.plot_pairs();



.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_68_0.png


Note that if working in a non-interactive environment, you can use e.g.
``plt.savefig('pairs.png')`` after an ELFI plotting command to save the
current figure to disk.

Sequential Monte Carlo ABC
--------------------------

Rejection sampling is quite inefficient, as it does not learn from its
history. The sequential Monte Carlo (SMC) ABC algorithm does just that
by applying importance sampling: samples are *weighed* according to the
resulting discrepancies and the next *population* of samples is drawn
near to the previous using the weights as probabilities.

For evaluating the weights, SMC ABC needs to be able to compute the
probability density of the generated parameters. In our MA2 example we
used custom priors, so we have to specify a ``pdf`` function by
ourselves. If we used standard priors, this step would not be needed.
Let's modify the prior distribution classes:

.. code:: ipython3

    # define prior for t1 as in Marin et al., 2012 with t1 in range [-b, b]
    class CustomPrior_t1(elfi.Distribution):
        def rvs(b, size=1, random_state=None):
            u = scipy.stats.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
            t1 = np.where(u<0.5, np.sqrt(2.*u)*b-b, -np.sqrt(2.*(1.-u))*b+b)
            return t1
        
        def pdf(x, b):
            p = 1./b - np.abs(x) / (b*b)
            p = np.where(p < 0., 0., p)  # disallow values outside of [-b, b] (affects weights only)
            return p
    
        
    # define prior for t2 conditionally on t1 as in Marin et al., 2012, in range [-a, a]
    class CustomPrior_t2(elfi.Distribution):
        def rvs(t1, a, size=1, random_state=None):
            locs = np.maximum(-a-t1, t1-a)
            scales = a - locs
            t2 = scipy.stats.uniform.rvs(loc=locs, scale=scales, size=size, random_state=random_state)
            return t2
        
        def pdf(x, t1, a):
            locs = np.maximum(-a-t1, t1-a)
            scales = a - locs
            p = scipy.stats.uniform.pdf(x, loc=locs, scale=scales)
            p = np.where(scales>0., p, 0.)  # disallow values outside of [-a, a] (affects weights only)
            return p
        
        
    # Redefine the priors
    t1.become(elfi.Prior(CustomPrior_t1, 2, model=t1.model))
    t2.become(elfi.Prior(CustomPrior_t2, t1, 1))

Run SMC ABC
~~~~~~~~~~~

In ELFI, one can setup a SMC ABC sampler just like the Rejection
sampler:

.. code:: ipython3

    smc = elfi.SMC(d, batch_size=10000, seed=seed)

For sampling, one has to define the number of output samples, the number
of populations and a *schedule* i.e. a list of quantiles to use for each
population. In essence, a population is just refined rejection sampling.

.. code:: ipython3

    N = 1000
    schedule = [0.7, 0.2, 0.05]
    %time result_smc = smc.sample(N, schedule)


.. parsed-literal::

    INFO:elfi.methods.parameter_inference:---------------- Starting round 0 ----------------
    INFO:elfi.methods.parameter_inference:---------------- Starting round 1 ----------------
    INFO:elfi.methods.parameter_inference:---------------- Starting round 2 ----------------


.. parsed-literal::

    CPU times: user 1.36 s, sys: 241 ms, total: 1.6 s
    Wall time: 1.62 s


We can have summaries and plots of the results just like above:

.. code:: ipython3

    result_smc.summary


.. parsed-literal::

    Method: SMC
    Number of posterior samples: 1000
    Number of simulations: 190000
    Threshold: 0.0492
    Posterior means for final population: t1: 0.552, t2: 0.205


The ``Sample`` object returned by the SMC-ABC sampling contains also
some methods for investigating the evolution of populations, e.g.:

.. code:: ipython3

    result_smc.posterior_means_all_populations


.. parsed-literal::

    Posterior means for population 0: t1: 0.547, t2: 0.232
    Posterior means for population 1: t1: 0.559, t2: 0.23
    Posterior means for population 2: t1: 0.552, t2: 0.205
    


.. code:: ipython3

    result_smc.plot_marginals_all_populations(bins=25, figsize=(8, 2), fontsize=12)



.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_81_0.png



.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_81_1.png



.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_81_2.png


Obviously one still has direct access to the samples as well, which
allows custom plotting:

.. code:: ipython3

    n_populations = len(schedule)
    fig, ax = plt.subplots(ncols=n_populations, sharex=True, sharey=True, figsize=(16,6))
    samples = [pop.samples_list for pop in result_smc.populations]
    for ii in range(n_populations):
        s = samples[ii]
        ax[ii].scatter(s[0], s[1], s=5, edgecolor='none');
        ax[ii].set_title("Population {}".format(ii));
        ax[ii].plot([0, 2, -2, 0], [-1, 1, 1, -1], 'b')
    ax[0].set_xlabel(result_smc.names_list[0]);
    ax[0].set_ylabel(result_smc.names_list[1]);
    ax[0].set_xlim([-2, 2])
    ax[0].set_ylim([-1, 1]);



.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/usage/tutorial_files/tutorial_83_0.png


It can be seen that the populations iteratively concentrate more and
more around the true parameter values.

Note that for the later populations some of the samples lie outside
allowed region. This is due to the SMC algorithm sampling near previous
samples, with *near* meaning a Gaussian distribution centered around
previous samples with variance as twice the weighted empirical variance.
However, the outliers carry zero weight, and have no effect on the
estimates.

That's it! See the other documentation for more advanced topics on e.g.
BOLFI, external simulators and parallelization.

