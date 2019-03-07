Changelog
=========


0.7.4 (2019-03-07)
------------------
- Add sampler option `algorithm` for bolfi-posterior-sampling
- Add a check whether the option given for `algorithm` is one if the implemented samplers
- Add metropolis sampler `algorithm=metropolis` for bolfi-posterior-sampling
- Add option `warmup` to metropolis-sampler
- Add a small test of metropolis-sampler
- Fix bug in plot_discrepancy for more than 6 parameters
- Implement plot_gp for BayesianOptimization classes for plotting discrepancies
  and pair-wise contours in case when we have arbitrary number of parameters
- Fix lint

0.7.3 (2018-08-30)
------------------
- Fix bug in plot_pairs which crashes in case of 1 parameter
- Fix bug in plot_marginals which outputs empty plots in case where we have parameter more than 5
- Fix crashing summary and plots for samples with multivariate priors
- Add progress bar for inference methods
- Add method save to Sample objects
- Add support for giving seed to `generate`
- Implement elfi.plot_params_vs_node for plotting parameters vs. node output

0.7.2 (2018-06-20)
------------------
- Added support for kwargs in elfi.set_client
- Added new example: inference of transmission dynamics of bacteria in daycare centers
- Added new example: Lorenz model

0.7.1 (2018-04-11)
------------------
- Implemented model selection (elfi.compare_models). See API documentation.
- Fix threshold=0 in rejection sampling
- Set default batch_size to 1 in ParameterInference base class

0.7 (2017-11-30)
----------------
- Added new example: the stochastic Lotka-Volterra model
- Fix methods.bo.utils.minimize to be strictly within bounds
- Implemented the Two Stage Procedure, a method of summary-statistics diagnostics
- Added the MaxVar acquisition method
- Added the RandMaxVar acquisition method
- Added the ExpIntVar acquisition method
- Implemented the Two Stage Procedure, a method of summary-statistics diagnostics
- Added new example: the stochastic Lotka-Volterra model
- Fix methods.bo.utils.minimize to be strictly within bounds
- Fix elfi.Distance to support scipy 1.0.0

0.6.3 (2017-09-28)
------------------

- Further performance improvements for rerunning inference using stored data via caches
- Added the general Gaussian noise example model (fixed covariance)
- restrict NetworkX to versions < 2.0

0.6.2 (2017-09-06)
------------------

- Easier saving and loading of ElfiModel
- Renamed elfi.set_current_model to elfi.set_default_model
- Renamed elfi.get_current_model to elfi.get_default_model
- Improved performance when rerunning inference using stored data
- Change SMC to use ModelPrior, use to immediately reject invalid proposals

0.6.1 (2017-07-21)
------------------

- Fix elfi.Prior and NoneType error #203
- Fix a bug preventing the reuse of ArrayPool data with a new inference
- Added pickling for OutputPool:s
- Added OutputPool.open to read a closed pool from disk
- Refactored Sample and SmcSample classes
- Added elfi.new_model method
- Made elfi.set_client method to accept clients as strings for easier client switching
- Fixed a bug in NpyArray that would lead to an inconsistent state if multiple
  simultaneous instances were opened.
- Added the ability to move the pool data folder
- Sample.summary is now a method instead of a property
- SmcSample methods takes the keyword argument 'all' to show results of all populations
- Added a section about iterative advancing to documentation

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
