ELFI - Engine for Likelihood-Free Inference
===========================================

ELFI is a statistical software package written in Python for Approximative Bayesian Computation (ABC_), also known e.g. as likelihood-free inference, simulator-based inference, approximative Bayesian inference etc. This is useful, when the likelihood function is unknown or difficult to evaluate, but a generative simulator model exists.

.. _ABC: https://en.wikipedia.org/wiki/Approximate_Bayesian_computation

The probabilistic inference model is defined as a directed acyclic graph, which allows for an intuitive means to describe inherent dependencies in the model. The inference pipeline is automatically parallelized from multiple cores up to a cluster environment. ELFI also handles seeding the random number generators and storing of the generated data for you so that you can easily repeat or fine tune your inference. Additionally, the package includes functionality for visualization.

Currently implemented ABC methods:

- rejection sampler
- Sequential Monte Carlo ABC sampler
- Bayesian Optimization for Likelihood-Free Inference (BOLFI_) framework

.. _BOLFI: http://jmlr.csail.mit.edu/papers/v17/15-017.html

Other notable included algorithms and methods:
- Bayesian Optimization
- No-U-Turn-Sampler_, a Hamiltonian Monte Carlo MCMC sampler

.. _No-U-Turn-Sampler: http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf

GitHub page: https://github.com/elfi-dev/elfi

See examples under the notebooks_ directory to get started. Limited user-support may be asked from elfi-support.at.hiit.fi, but the `Gitter chat`_ is preferable.

.. _notebooks: https://github.com/elfi-dev/notebooks
.. _Gitter chat: https://gitter.im/elfi-dev/elfi?utm_source=share-link&utm_medium=link&utm_campaign=share-link

Licenses:

- Code: BSD3_
- Documentation: `CC-BY 4.0`_

.. _BSD3: https://opensource.org/licenses/BSD-3-Clause
.. _CC-BY 4.0: https://creativecommons.org/licenses/by/4.0

