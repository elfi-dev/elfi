ELFI - Engine for Likelihood-Free Inference
===========================================

<!-- .. image:: https://img.shields.io/pypi/v/elfi.svg
        :target: https://pypi.python.org/pypi/elfi

.. image:: https://img.shields.io/travis/HIIT/elfi.svg
        :target: https://travis-ci.com/HIIT/elfi

.. image:: https://readthedocs.org/projects/elfi/badge/?version=latest
        :target: https://elfi.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
 
 https://github.com/dwyl/repo-badges
 -->

[![Build Status](https://travis-ci.com/HIIT/elfi.svg?token=xAu1DN2J4WjCapVWLinn&branch=dev)](https://travis-ci.com/HIIT/elfi)
[![Documentation Status](https://readthedocs.org/projects/elfi/badge/?version=latest)](http://elfi.readthedocs.io/en/latest/?badge=latest)

<img src="https://cloud.githubusercontent.com/assets/1233418/20178983/6e22ee44-a75c-11e6-8345-5934b55b9dc6.png" width="15%" align="right"></img>

ELFI is a statistical software package written in Python for Approximative Bayesian Computation ([ABC](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation)), also known as likelihood-free inference, simulator-based inference, approximative Bayesian inference etc. This is useful, when the likelihood function is unknown or difficult to evaluate, but a generative simulator model exists.

The probabilistic inference model is defined as a directed acyclic graph, which allows for an intuitive means to describe inherent dependencies in the model. The inference pipeline is automatically parallelized with [Dask](https://dask.pydata.org), which scales well from a desktop up to a cluster environment. The package includes functionality for input/output operations and visualization.

Currently implemented ABC methods:
- rejection sampler
- sequential Monte Carlo sampler
- [Bayesian Optimization for Likelihood-Free Inference (BOLFI) framework](http://jmlr.csail.mit.edu/papers/v17/15-017.html)

See examples under [notebooks](notebooks) to get started. Full documentation can be found at http://elfi.readthedocs.io/. Limited user-support may be asked from elfi-support.at.hiit.fi.

<!-- ..
   Installation
   -------------
   ::

     pip install elfi
 -->

Developer installation
----------------------
```
git clone https://github.com/HIIT/elfi.git
cd elfi
pip install numpy
pip install -r requirements-dev.txt
pip install -e .
```

It is recommended to create a virtual environment for development before installing.

Virtual environment using Anaconda
----------------------------------
Below an example how to create a virtual environment named ``elfi`` using Anaconda:

    conda create -n elfi python=3* scipy

Then activate it:

    source activate elfi
