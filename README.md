ELFI - Engine for Likelihood-Free Inference
===========================================

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


Installation
------------
```
pip install elfi
```

ELFI is currently tested only with Python 3.5. If you are new to Python, perhaps the simplest way to install it is [Anaconda](https://www.continuum.io/downloads).

Currently it is required to use Distributed 1.14.3.

Virtual environment using Anaconda
----------------------------------
If you want to create a virtual environment before installing, you can do so with Anaconda:

```
conda create -n elfi python=3.5 scipy
source activate elfi
pip install elfi
```

Potential problems with installation
------------------------------------
ELFI depends on several other Python packages, which have their own dependencies. Resolving these may sometimes go wrong:
- If you receive an error about missing `numpy`, please install it first.
- If you receive an error about `yaml.load`, install `pyyaml`.
- On OS X with Anaconda virtual environment say `conda install python.app` and then use `pythonw` instead of `python`.
- Note that ELFI currently supports Python 3.5 only, although 3.x may work as well.
