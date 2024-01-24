Changelog
=========

0.8.7 (2023-09-21)
------------------
- Update available methods list
- Pin `numpy <= 1.24.0` due to `GPy`
- Update RTD configuration

0.8.6 (2023-03-07)
------------------
- Fix broken 0.8.5 by adding missing `__init__.py` to elfi/methods/bsl/

0.8.5 (2023-03-07)
------------------
- Fix the option to continue inference in model-based inference
- Move classifiers
- Fix readthedocs configuration
- Update penalty to shrinkage parameter conversion in synthetic likelihood calculation
- Update BSL pre sample methods
- Update BslSample
- Move ROMC tests
- Update to numpy 1.24
- Update readthedocs configuration
- Restrict numpy < 1.24 until codebase has been updated
- Update documentation-related files: docs, conf.py and requirements-dev.txt 
- Update PULL_REQUEST_TEMPLATE.md
- Drop tests for py36 and add tests for py39 and py310
- Fix couple of minor bugs in `ar1`-model
- Update parent class in BOLFIRE
- Fix semiparametric synthetic likelihood with glasso/warton and add tests
- Fix plot marginals and remove plot summaries
- Fix stochastic volatility example
- Improve batch simulations in toad example
- Remove synthetic likelihood node and update BSL data collection
- Fix M/G/1 example
- Fix scratch assay example
- Add scratch assay example
- Add GP classifier for ratio estimation
- Fix multidimensional indexing in daycare example
- Add BSL method

0.8.4 (2022-06-13)
------------------
- Modify Lotka-Volterra model's priors as many methods do not support discrete random variables.
- Fix acquisition index in state plot
- Reformat `summary()` for `Sample(ParameterInferenceResult)`
- Fix linting in `arch.py`
- Add summary statistics to Lotka-Volterra model
- Add boolean `observation_noise` option to `lotka_volterra`
- Add parameter names as an optional input in model prior and fix the parameter order in priors used in BOLFI and BOLFIRE
- Add feature names as an optional input and make training data size a required input in BOLFIRE
- Fix the observed property in simulator nodes
- Fix default outputs in generate
- Add docstring description to ARCH-model
- Make MAP estimates calculation in BOLFIRE optional and based on log-posterior
- Use batch system to run simulations in BOLFIRE
- Use `target_model.parameter_names` from instead of `model.parameter_names` in `BOLFIRE`
- Extract BO results using `target_model.parameter_names` from instead of `model.parameter_names`
- Update tox.ini
- Add option to use additive acquisition cost in LCBSC
- Change sigma_proposals-input in metropolis from list to dict
- Fix is_array in utils
- Fix acq_noise_var-bug in acquisition.py. Influenced BOLFI.

0.8.3 (2022-02-17)
------------------
- Add a new inference method: BOLFIRE
- Fix the hessian approximation, visualizations and the line search algorithm in ROMC
- Add tests for all ROMC parts

0.8.2 (2021-10-13)
------------------
- Relax tightly pinned dependency on a version of dask[distributed]
- Change lotka-volterra priors to follow the given reference
- Fix README.md badges
- Fix a few small issues with CONTRIBUTING.rst
- Add Github Actions based CI workflow
- Add the skeleton of TestBench-functionality for comparing methods
- Fix a bug of plot_traces() not working if there is only 1 chain 
- Fix histograms in pair_plot diagonals and improve visual outlook
- Improve axes creation and visual outlook
- Fix a bug where precomputed evidence size was not taken into account when reporting BOLFI-results
- Fix a bug where observable nodes were not colored gray when using `elfi.draw`
- Add `plot_predicted_node_pairs` in visualization.py.

0.8.0 (2021-03-29)
------------------
- Merge adaptive distance ABC-SMC and ABC-SMC functionalities
- Split `DensityRatioEstimation` from utils.py into separate file
- Refactor parameter_inferency.py into methodtype-wise individual files  
- Rename `elfi.methods.mcmc.gelman_rubin` as `elfi.methods.mcmc.gelman_rubin_statistic`
- Refactor `class ModelPrior` from `methods.utils` to `model.extensions`.
- Add adaptive threshold selection method for ABC-SMC
- Modify ProgressBar-functionality
- Add constrains to ExpIntVar-acquisition so that no queries will be outside prior support
- Add ABC-SMC with adaptive distance
- Add Robust optimisation Monte Carlo method
- Fix small issues in ABC-SMC which did not work in 1-dimensional problems or with output names
- Update README.md

0.7.7 (2020-10-12)
------------------
- Update info to reflect setting python 3.6 as the default version
- Update documentation to setting python 3.6 as default
- Add dask support to elfi client options
- Add python 3.7 to travis tests and remove python 3.5 due to clash with dask
- Modify progress bar to better indicate ABC-SMC inference status
- Change networkx support from 1.X to 2.X
- Improve docstrings in elfi.methods.bo.acquisition
- Fix readthedocs-build by adding .readthedocs.yml and restricting the build to
  python3.5, for now

0.7.6 (2020-08-29)
------------------
- Fix incompatibility with scipy>1.5 in bo.utils.stochastic_optimization
- Minor improvements to documentation

0.7.5 (2019-12-18)
------------------
- Improved the appearance of figures produced by `plot_gp` and added the option
  to draw true parameter indicators on the subplots using the optional input
  `true_params`
- Modified DCC model by taking into account that subject can't infect herself
- Added ability to set minimizer constrains for BOLFI
- Enable bolfi.fit using only pre-generated initial evidence points
- Fixed a bug causing random seed number to be deterministic
- Updated requirements-dev.txt with pytest>=4.4
- Minor changes to documentation and refactoring
- Added `make test-notslow` alternative

0.7.4 (2019-03-07)
------------------
- Add sampler option `algorithm` for bolfi-posterior-sampling
- Add a check whether the option given for `algorithm` is one if the
  implemented samplers
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
- Fix bug in plot_marginals which outputs empty plots in case where we have
  parameter more than 5
- Fix crashing summary and plots for samples with multivariate priors
- Add progress bar for inference methods
- Add method save to Sample objects
- Add support for giving seed to `generate`
- Implement elfi.plot_params_vs_node for plotting parameters vs. node output

0.7.2 (2018-06-20)
------------------
- Added support for kwargs in elfi.set_client
- Added new example: inference of transmission dynamics of bacteria in daycare
  centers
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
- Implemented the Two Stage Procedure, a method of summary-statistics
  diagnostics
- Added the MaxVar acquisition method
- Added the RandMaxVar acquisition method
- Added the ExpIntVar acquisition method
- Implemented the Two Stage Procedure, a method of summary-statistics
  diagnostics
- Added new example: the stochastic Lotka-Volterra model
- Fix methods.bo.utils.minimize to be strictly within bounds
- Fix elfi.Distance to support scipy 1.0.0

0.6.3 (2017-09-28)
------------------

- Further performance improvements for rerunning inference using stored data
  via caches
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
- Made elfi.set_client method to accept clients as strings for easier client
  switching
- Fixed a bug in NpyArray that would lead to an inconsistent state if multiple
  simultaneous instances were opened.
- Added the ability to move the pool data folder
- Sample.summary is now a method instead of a property
- SmcSample methods takes the keyword argument 'all' to show results of all
  populations
- Added a section about iterative advancing to documentation

0.6 (2017-07-03)
----------------

- Changed some of the internal variable names in methods.py. Most notable
  outputs is now
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
- scheduler-independent parallelization interface (currently supports native &
  ipyparallel)
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
