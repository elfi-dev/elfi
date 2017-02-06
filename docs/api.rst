*************
API Reference
*************

Graph
^^^^^
.. currentmodule:: elfi.graph
.. autosummary::

   Graph
   Node
   
.. automodule:: elfi.graph
   :members:

Inference context
^^^^^^^^^^^^^^^^^
.. currentmodule:: elfi.inference_task
.. autosummary::

   InferenceTask
   
.. automodule:: elfi.graph
   :members:

Distributions
^^^^^^^^^^^^^
.. currentmodule:: elfi.distributions
.. autosummary::

   Distribution
   ScipyDistribution
   RandomVariable
   SMCProposal
   
.. automodule:: elfi.distributions
   :members:

Wrapping external simulators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: elfi.wrapper
.. autosummary::

   Wrapper
   
.. automodule:: elfi.wrapper
   :members:

Inference methods
^^^^^^^^^^^^^^^^^
.. currentmodule:: elfi.methods
.. autosummary::

   ABCMethod
   Rejection
   BOLFI
   
.. automodule:: elfi.methods
   :members:

Bayesian optimization
^^^^^^^^^^^^^^^^^^^^^

Surrogate models
~~~~~~~~~~~~~~~~
.. currentmodule:: elfi.bo.gpy_model
.. autosummary::

   GPyModel
   
.. automodule:: elfi.bo.gpy_model
   :members:

Acquisition functions
~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: elfi.bo.acquisition
.. autosummary::

   AcquisitionBase
   AcquisitionSchedule
   LCBAcquisition
   RandomAcquisition
   
.. automodule:: elfi.bo.acquisition
   :members:

Results
^^^^^^^
.. currentmodule:: elfi.result
.. autosummary::

   Result
   Result_SMC
   Result_BOLFI
   
.. automodule:: elfi.result
   :members:

Visualization
^^^^^^^^^^^^^
.. currentmodule:: elfi.visualization
.. autosummary::

   draw_model
   plot_marginals
   plot_pairs
   
.. automodule:: elfi.visualization
   :members:

Persistence
^^^^^^^^^^^

Interfaces
~~~~~~~~~~
.. currentmodule:: elfi.storage
.. autosummary::

   ElfiStore
   LocalElfiStore
   LocalDataStore
   NameIndexDataInterface
   SerializedStoreInterface

Implementations
~~~~~~~~~~~~~~~
.. currentmodule:: elfi.storage
.. autosummary::

   MemoryStore
   DictListStore
   UnQLiteStore
   
.. automodule:: elfi.storage
   :members:

Core
^^^^
.. currentmodule:: elfi.core
.. autosummary::

   Operation
   Constant
   Simulator
   Summary
   Discrepancy
   Transform
   
.. automodule:: elfi.core
   :members:
