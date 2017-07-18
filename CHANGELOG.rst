Changelog
==========

dev
---

- Fix elfi.Prior and NoneType error #203
- Fix a bug preventing the reuse of ArrayPool data with a new inference
- Added a possibility to pickle ArrayPool
- Added ArrayPool.open to read a closed pool from disk
- Refactored Sample and SmcSample classes
- Added elfi.new_model method
- Made elfi.set_client method to accept clients as strings.


0.6 (2017-07-03)
----------------

- Changed some of the internal variable names in methods.py. Most notable outputs is now
  output_names.
- methods.py renamed to parameter_inference.py
- Changes in elfi.methods.results module class names:
  - OptimizationResult (a new result type)
  - Result -> Sample
  - ResultSMC -> SmcSample
  - ResultBOLFI -> BolfiSample
- Changes in BO/BOLFI:
  - take advantage of priors
  - take advantage of seed
  - improved optimization scheme
  - bounds must be a dict
- two new toy examples added: Gaussian and the Ricker model

0.5 (2017-05-19)
----------------

Major update, a lot of code base rewritten.

Most important changes:

- revised syntax for model definition (esp. naming)
- scheduler-independent parallelization interface (currently supports native & ipyparallel)
- methods can now be run iteratively
- persistence to .npy files
- Bayesian optimization as a separate method
- sampling in BOLFI
- MCMC sampling using the No-U-Turn-Sampler (NUTS)
- Result object for BOLFI
- virtual vectorization of external operations

See the updated notebooks and documentation for examples and details.

0.3.1 (2017-01-31)
------------------

- Clean up requirements
- Set graphviz and unqlite optional
- PyPI release (pip install elfi)

0.2.2 - 0.3
-----------

- The inference problem is now contained in an Inference Task object.
- SMC-ABC has been reimplemented.
- Results from inference are now contained in a Result object.
- Integrated basic visualization.
- Added a notebook demonstrating usage with external simulators and operations.
- Lot's of refactoring and other minor changes.
