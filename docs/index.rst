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

Additionally, ELFI integrates tools for visualization, model comparison, diagnostics and post-processing.


.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
    good-to-know
    quickstart
    api
    faq

.. toctree::
    :maxdepth: 1
    :caption: Usage

    usage/tutorial
    usage/parallelization
    usage/BOLFI
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


Citation
--------

If you wish to cite ELFI, please use the paper in JMLR_:

.. _JMLR: http://www.jmlr.org/papers/v19/17-374.html

.. code-block:: console

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
