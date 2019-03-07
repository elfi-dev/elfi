**Version 0.7.4 released!** See the CHANGELOG and [notebooks](https://github.com/elfi-dev/notebooks).

**NOTE:** For the time being NetworkX 2 is incompatible with ELFI.

ELFI - Engine for Likelihood-Free Inference
===========================================

[![Build Status](https://travis-ci.org/elfi-dev/elfi.svg?branch=master)](https://travis-ci.org/elfi-dev/elfi)
[![Code Health](https://landscape.io/github/elfi-dev/elfi/dev/landscape.svg?style=flat)](https://landscape.io/github/elfi-dev/elfi/dev)
[![Documentation Status](https://readthedocs.org/projects/elfi/badge/?version=latest)](http://elfi.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/elfi-dev/elfi.svg)](https://gitter.im/elfi-dev/elfi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![DOI](https://zenodo.org/badge/69855441.svg)](https://zenodo.org/badge/latestdoi/69855441)

<img src="https://cloud.githubusercontent.com/assets/1233418/20178983/6e22ee44-a75c-11e6-8345-5934b55b9dc6.png" width="15%" align="right"></img>

ELFI is a statistical software package written in Python for likelihood-free inference (LFI) such as Approximate
Bayesian Computation ([ABC](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation)).
The term LFI refers to a family of inference methods that replace the use of the likelihood function with a data
generating simulator function. ELFI features an easy to use generative modeling syntax and supports parallelized
inference out of the box.

Currently implemented LFI methods:
- ABC Rejection sampler
- Sequential Monte Carlo ABC sampler
- [Bayesian Optimization for Likelihood-Free Inference (BOLFI)](http://jmlr.csail.mit.edu/papers/v17/15-017.html)

Other notable included algorithms and methods:
- Bayesian Optimization
- [No-U-Turn-Sampler](http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf), a Hamiltonian Monte Carlo MCMC sampler

ELFI also integrates tools for visualization, model comparison, diagnostics and post-processing.

See examples under [notebooks](https://github.com/elfi-dev/notebooks) to get started. Full
documentation can be found at http://elfi.readthedocs.io/. Limited user-support may be
asked from elfi-support.at.hiit.fi, but the
[Gitter chat](https://gitter.im/elfi-dev/elfi?utm_source=share-link&utm_medium=link&utm_campaign=share-link)
is preferable.


Installation
------------

ELFI requires Python 3.5 or greater. You can install ELFI by typing in your terminal:

```
pip install elfi
```
or on some platforms using Python 3 specific syntax:
```
pip3 install elfi
```

Note that in some environments you may need to first install `numpy` with
`pip install numpy`. This is due to our dependency to `GPy` that uses `numpy` in its installation.

### Optional dependencies

- `graphviz` for drawing graphical models (needs [Graphviz](http://www.graphviz.org)), highly recommended


### Installing Python 3

If you are new to Python, perhaps the simplest way to install a specific version of Python
is with [Anaconda](https://www.continuum.io/downloads).

### Virtual environment using Anaconda

It is very practical to create a virtual Python environment. This way you won't interfere
with your default Python environment and can easily use different versions of Python
in different projects. You can create a virtual environment for ELFI using anaconda with:

```
conda create -n elfi python=3.5 numpy
source activate elfi
pip install elfi
```

### Docker container

A simple Dockerfile with Jupyter support is also provided. This is especially suitable for running tests. Please see [Docker documentation](https://docs.docker.com/) for details.

```
git clone --depth 1 https://github.com/elfi-dev/elfi.git
cd elfi
make docker-build  # builds the image with requirements for dev
make docker  # runs a container with live elfi directory
```

To open a Jupyter notebook, run
```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
within the container and then on host open the page http://localhost:8888. 

### Potential problems with installation

ELFI depends on several other Python packages, which have their own dependencies.
Resolving these may sometimes go wrong:
- If you receive an error about missing `numpy`, please install it first.
- If you receive an error about `yaml.load`, install `pyyaml`.
- On OS X with Anaconda virtual environment say `conda install python.app` and then use
`pythonw` instead of `python`.
- Note that ELFI requires Python 3.5 or greater so try `pip3 install elfi`.
- Make sure your Python installation meets the versions listed in `requirements.txt`.


Citation
--------

If you wish to cite ELFI, please use the paper in [JMLR](http://www.jmlr.org/papers/v19/17-374.html):

```
@article{JMLR:v19:17-374,
  author  = {Jarno Lintusaari and Henri Vuollekoski and Antti Kangasr{\"a}{\"a}si{\"o} and Kusti Skyt{\'e}n and Marko J{\"a}rvenp{\"a}{\"a} and Pekka Marttinen and Michael U. Gutmann and Aki Vehtari and Jukka Corander and Samuel Kaski},
  title   = {ELFI: Engine for Likelihood-Free Inference},
  journal = {Journal of Machine Learning Research},
  year    = {2018},
  volume  = {19},
  number  = {16},
  pages   = {1-7},
  url     = {http://jmlr.org/papers/v19/17-374.html}
}
```
