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
- ABC-SMC sampler with `adaptive distance`_
- ABC-SMC sampler with `adaptive threshold selection`_
- Bayesian Optimization for Likelihood-Free Inference (BOLFI_) framework
- Robust Optimization Monte Carlo (ROMC_) framework
- Bayesian Optimization for Likelihood-Free Inference by Ratio Estimation (BOLFIRE_)
- Bayesian Synthetic Likelihood (BSL_)

.. _adaptive distance: https://projecteuclid.org/euclid.ba/1460641065
.. _adaptive threshold selection: https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Adaptive-Approximate-Bayesian-Computation-Tolerance-Selection/10.1214/20-BA1211.full
.. _BOLFI: http://jmlr.org/papers/v17/15-017.html
.. _ROMC:  http://proceedings.mlr.press/v108/ikonomov20a.html
.. _BOLFIRE: https://helda.helsinki.fi/handle/10138/305039
.. _BSL: https://doi.org/10.1080/10618600.2017.1302882

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
    usage/adaptive_distance
    usage/adaptive_threshold_selection
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
