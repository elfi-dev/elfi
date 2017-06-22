.. ELFI documentation master file. It should at least contain the root `toctree` directive.


ELFI - Engine for Likelihood-Free Inference
===========================================

.. include:: description.rst

.. image:: http://research.cs.aalto.fi/pml/software/elfi/docs/0.5/images/ma2.png
   :alt: MA2 model in ELFI
   :align: right

See :doc:`the quickstart <quickstart>` to get started.

ELFI is licensed under BSD3_. The source is in GitHub_.

.. _BSD3: https://opensource.org/licenses/BSD-3-Clause
.. _GitHub: https://github.com/elfi-dev/elfi


Currently implemented LFI methods:
----------------------------------

- ABC rejection sampler
- Sequential Monte Carlo ABC sampler
- Bayesian Optimization for Likelihood-Free Inference (BOLFI_) framework

.. _BOLFI: http://jmlr.org/papers/v17/15-017.html

ELFI also has the following non LFI methods:

- Bayesian Optimization
- No-U-Turn-Sampler_, a Hamiltonian Monte Carlo MCMC sampler

.. _No-U-Turn-Sampler: http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf


.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
    good-to-know
    quickstart
    api

.. toctree::
    :maxdepth: 1
    :caption: Usage

    usage/tutorial
    usage/parallelization
    usage/external
    usage/implementing-methods

.. toctree::
    :maxdepth: 1
    :caption: Developer documentation

    developer/architecture
    developer/contributing

..    faq

..    usage/implementing-methods

..    developer/architecture
..    developer/extensions




Other names or related approaches to LFI include simulator-based inference, approximate
Bayesian inference, indirect inference, etc.