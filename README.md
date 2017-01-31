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

[![Build Status](https://travis-ci.org/HIIT/elfi.svg?branch=master)](https://travis-ci.org/HIIT/elfi)
[![Code Health](https://landscape.io/github/HIIT/elfi/master/landscape.svg?style=flat)](https://landscape.io/github/HIIT/elfi/master)
[![Documentation Status](https://readthedocs.org/projects/elfi/badge/?version=latest)](http://elfi.readthedocs.io/en/latest/?badge=latest)
[![Gitter chat](https://badges.gitter.im/HIIT/elfi.svg)](https://gitter.im/HIIT/elfi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

<img src="https://cloud.githubusercontent.com/assets/1233418/20178983/6e22ee44-a75c-11e6-8345-5934b55b9dc6.png" width="15%" align="right"></img>

ELFI is a statistical software package written in Python for Approximative Bayesian Computation ([ABC](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation)), also known as likelihood-free inference, simulator-based inference, approximative Bayesian inference etc. This is useful, when the likelihood function is unknown or difficult to evaluate, but a generative simulator model exists.

The probabilistic inference model is defined as a directed acyclic graph, which allows for an intuitive means to describe inherent dependencies in the model. The inference pipeline is automatically parallelized with [Dask](https://dask.pydata.org), which scales well from a desktop up to a cluster environment. The package includes functionality for input/output operations and visualization.

Currently implemented ABC methods:
- rejection sampler
- sequential Monte Carlo sampler
- [Bayesian Optimization for Likelihood-Free Inference (BOLFI) framework](http://jmlr.csail.mit.edu/papers/v17/15-017.html)

See examples under [notebooks](notebooks) to get started. Full documentation can be found at http://elfi.readthedocs.io/. Limited user-support may be asked from elfi-support.at.hiit.fi, but the [Gitter chat](https://gitter.im/HIIT/elfi?utm_source=share-link&utm_medium=link&utm_campaign=share-link) is preferable.

<!-- ..
   Installation
   -------------
   ::

     pip install elfi
 -->

Developer installation
----------------------
ELFI is currently tested only with Python 3.5. If you are new to Python, perhaps the simplest way to install it is [Anaconda](https://www.continuum.io/downloads).

Currently we recommend using Distributed 1.14.3.
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

    conda create -n elfi python=3.5 scipy

Then activate it:

    source activate elfi
