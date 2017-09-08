API
===

This file describes the classes and methods available in ELFI.

Modelling API
-------------
Below is the API for creating generative models.

.. autosummary::
   elfi.ElfiModel

**General model nodes**

.. autosummary::
   elfi.Constant
   elfi.Operation
   elfi.RandomVariable

**LFI nodes**

.. autosummary::
   elfi.Prior
   elfi.Simulator
   elfi.Summary
   elfi.Discrepancy
   elfi.Distance

**Other**

.. currentmodule:: elfi.model.elfi_model

.. autosummary::
   elfi.new_model
   elfi.load_model
   elfi.get_default_model
   elfi.set_default_model

.. currentmodule:: elfi.visualization.visualization

.. autosummary::
   elfi.draw

Inference API
-------------

Below is a list of inference methods included in ELFI.

.. autosummary::
   elfi.Rejection
   elfi.SMC
   elfi.BayesianOptimization
   elfi.BOLFI

**Empirical density estimation**

.. autosummary::
   elfi.distributions.EmpiricalDensity
   elfi.distributions.ecdf
   elfi.distributions.eppf
   elfi.distributions.MetaGaussian

**Result objects**

.. currentmodule:: elfi.methods.results

.. autosummary::
   OptimizationResult
   Sample
   SmcSample
   BolfiSample

**Post-processing**

.. currentmodule:: elfi

.. autosummary::
   elfi.adjust_posterior

.. currentmodule:: elfi.methods.post_processing

.. autosummary::
   LinearAdjustment

Other
-----

**Data pools**

.. autosummary::
   elfi.OutputPool
   elfi.ArrayPool


**Module functions**

.. currentmodule:: elfi

.. autosummary::
   elfi.get_client
   elfi.set_client


**Tools**

.. currentmodule:: elfi.model.tools

.. autosummary::
   elfi.tools.vectorize
   elfi.tools.external_operation


Class documentations
--------------------

Modelling API classes
.....................

.. autoclass:: elfi.ElfiModel
   :members:

.. autoclass:: elfi.Constant
   :members:
   :inherited-members:

.. autoclass:: elfi.Operation
   :members:
   :inherited-members:

.. autoclass:: elfi.RandomVariable
   :members:
   :inherited-members:

.. autoclass:: elfi.Prior
   :members:
   :inherited-members:

.. autoclass:: elfi.Simulator
   :members:
   :inherited-members:

.. autoclass:: elfi.Summary
   :members:
   :inherited-members:

.. autoclass:: elfi.Discrepancy
   :members:
   :inherited-members:

.. autoclass:: elfi.Distance
   :members:
   :inherited-members:


**Other**

.. currentmodule:: elfi.model.elfi_model

.. automethod:: elfi.new_model

.. automethod:: elfi.get_current_model

.. automethod:: elfi.set_current_model

.. currentmodule:: elfi.visualization.visualization

.. automethod:: elfi.visualization.visualization.nx_draw

.. This would show undocumented members :undoc-members:


Inference API classes
.....................

.. autoclass:: elfi.Rejection
   :members:
   :inherited-members:

.. autoclass:: elfi.SMC
   :members:
   :inherited-members:

.. autoclass:: elfi.BayesianOptimization
   :members:
   :inherited-members:

.. autoclass:: elfi.BOLFI
   :members:
   :inherited-members:


**Empirical density estimation**

.. autoclass:: elfi.distributions.EmpiricalDensity
   :members:
   :inherited-members:

.. autoclass:: elfi.distributions.MetaGaussian
   :members:
   :inherited-members:

.. automethod:: elfi.distributions.ecdf
.. automethod:: elfi.distributions.eppf


.. currentmodule:: elfi.methods.results

**Result objects**

.. autoclass:: OptimizationResult
   :members:
   :inherited-members:

.. autoclass:: Sample
   :members:
   :inherited-members:

.. autoclass:: SmcSample
   :members:
   :inherited-members:

.. autoclass:: BolfiSample
   :members:
   :inherited-members:


**Post-processing**

.. currentmodule:: elfi

.. automethod:: elfi.adjust_posterior

.. currentmodule:: elfi.methods.post_processing

.. autoclass:: LinearAdjustment
   :members:
   :inherited-members:


Other
.....

**Data pools**

.. autoclass:: elfi.OutputPool
   :members:
   :inherited-members:

.. autoclass:: elfi.ArrayPool
   :members:
   :inherited-members:


**Module functions**

.. currentmodule:: elfi

.. automethod:: elfi.get_client

.. automethod:: elfi.set_client


**Tools**

.. currentmodule:: elfi.model.tools

.. automethod:: elfi.tools.vectorize

.. automethod:: elfi.tools.external_operation
