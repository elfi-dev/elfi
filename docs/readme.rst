ELFI - Engine for Likelihood-Free Inference
===========================================

ELFI is a statistical software package written in Python for Approximative Bayesian Computation (ABC_), also known e.g. as likelihood-free inference, simulator-based inference, approximative Bayesian inference etc. This is useful, when the likelihood function is unknown or difficult to evaluate, but a generative simulator model exists.

.. _ABC: https://en.wikipedia.org/wiki/Approximate_Bayesian_computation

The probabilistic inference model is defined as a directed acyclic graph, which allows for an intuitive means to describe inherent dependencies in the model. The inference pipeline is automatically parallelized with Dask_, which scales well from a desktop up to a cluster environment. The package includes functionality for input/output operations and visualization.

.. _Dask: https://dask.pydata.org

Currently implemented ABC methods:

- rejection sampler
- Sequential Monte Carlo sampler
- Bayesian Optimization for Likelihood-Free Inference (BOLFI_) framework

.. _BOLFI: https://arxiv.org/abs/1501.03291

GitHub page: https://github.com/HIIT/elfi

See examples under the notebooks directory to get started.

Licenses:

- Code: BSD3_
- Documentation: `CC-BY 4.0`_

.. _BSD3: https://opensource.org/licenses/BSD-3-Clause
.. _CC-BY 4.0: https://creativecommons.org/licenses/by/4.0

