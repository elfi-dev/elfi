API
===

This file describes the classes and methods available in ELFI.

Modelling API
-------------
Below is the API for creating generative models.

.. currentmodule:: .

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
   elfi.AdaptiveDistance

**Other**

.. autosummary::
   elfi.new_model
   elfi.load_model
   elfi.get_default_model
   elfi.set_default_model

.. autosummary::
   elfi.draw
   elfi.plot_params_vs_node

Inference API
-------------

Below is a list of inference methods included in ELFI.

.. autosummary::
   elfi.Rejection
   elfi.SMC
   elfi.AdaptiveDistanceSMC
   elfi.AdaptiveThresholdSMC
   elfi.BayesianOptimization
   elfi.BOLFI
   elfi.ROMC
   elfi.BSL


**Result objects**

.. currentmodule:: elfi.methods.results

.. autosummary::
   OptimizationResult
   Sample
   SmcSample
   BolfiSample


**Post-processing**

.. currentmodule:: .

.. autosummary::
   elfi.adjust_posterior

.. currentmodule:: elfi.methods.post_processing

.. autosummary::
   LinearAdjustment


**Diagnostics**

.. currentmodule:: .

.. autosummary::
   elfi.TwoStageSelection


**Acquisition methods**

.. currentmodule:: elfi.methods.bo.acquisition

.. autosummary::
   LCBSC
   MaxVar
   RandMaxVar
   ExpIntVar
   UniformAcquisition

Other
-----

**Data pools**

.. currentmodule:: .

.. autosummary::
   elfi.OutputPool
   elfi.ArrayPool


**Module functions**

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

.. autoclass:: elfi.AdaptiveDistance
   :members:
   :inherited-members:


**Other**

.. autofunction:: elfi.new_model

.. autofunction:: elfi.load_model

.. autofunction:: elfi.get_default_model

.. autofunction:: elfi.set_default_model

.. autofunction:: elfi.draw

.. autofunction:: elfi.plot_params_vs_node

.. This would show undocumented members :undoc-members:


Inference API classes
.....................

.. autoclass:: elfi.Rejection
   :members:
   :inherited-members:

.. autoclass:: elfi.SMC
   :members:
   :inherited-members:

.. autoclass:: elfi.AdaptiveDistanceSMC
   :members:
   :inherited-members:

.. autoclass:: elfi.AdaptiveThresholdSMC
   :members:
   :inherited-members:

.. autoclass:: elfi.BayesianOptimization
   :members:
   :inherited-members:

.. autoclass:: elfi.BOLFI
   :members:
   :inherited-members:

.. autoclass:: elfi.ROMC
   :members:
   :inherited-members:

.. autoclass:: elfi.BSL
   :members:
   :inherited-members:


**Result objects**

.. currentmodule:: elfi.methods.results

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

.. autoclass:: RomcSample
   :members:
   :inherited-members:

.. autoclass:: BslSample
   :members:
   :inherited-members:


**Post-processing**

.. currentmodule:: .

.. autofunction:: elfi.adjust_posterior

.. currentmodule:: elfi.methods.post_processing

.. autoclass:: LinearAdjustment
   :members:
   :inherited-members:

**Diagnostics**

.. currentmodule:: .

.. autoclass:: elfi.TwoStageSelection
   :members:
   :inherited-members:

**Acquisition methods**

.. currentmodule:: elfi.methods.bo.acquisition

.. autoclass:: LCBSC
   :members:
   :inherited-members:

.. autoclass:: MaxVar
   :members:
   :inherited-members:

.. autoclass:: RandMaxVar
   :members:
   :inherited-members:

.. autoclass:: ExpIntVar
   :members:
   :inherited-members:

.. autoclass:: UniformAcquisition
   :members:
   :inherited-members:

**Model selection**

.. currentmodule:: .

.. autofunction:: elfi.compare_models


Other
.....

**Data pools**

.. currentmodule:: .

.. autoclass:: elfi.OutputPool
   :members:
   :inherited-members:

.. autoclass:: elfi.ArrayPool
   :members:
   :inherited-members:


**Module functions**

.. autofunction:: elfi.get_client

.. autofunction:: elfi.set_client


**Tools**

.. currentmodule:: elfi.model.tools

.. automethod:: elfi.tools.vectorize

.. automethod:: elfi.tools.external_operation
