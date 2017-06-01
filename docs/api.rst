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


Inference API
-------------
Below is a list of inference methods included in ELFI.

.. autosummary::
   elfi.Rejection
   elfi.SMC
   elfi.BayesianOptimization
   elfi.BOLFI

**Result objects**

.. currentmodule:: elfi.methods.results

.. autosummary::
   Result
   ResultSMC
   ResultBOLFI

Class documentations
--------------------

Modelling API classes
.....................

.. autoclass:: elfi.ElfiModel
   :members:
   :inherited-members:

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

.. autoclass:: Result
   :members:
   :inherited-members:

.. autoclass:: ResultSMC
   :members:
   :inherited-members:

.. autoclass:: ResultBOLFI
   :members:
   :inherited-members: