ELFI
:::::

Engine for likelihood-free inference


Installation
============

Packages needed:

- numpy
- dask
- distributed

If you want the plotting then also:

- graphviz

You will probably need the latest version of dask and distributed::

  pip install git+https://github.com/dask/dask.git
  pip install git+https://github.com/dask/distributed.git

It is possible that I have forgotten something from this list, so
please modify this as needed.

Running the tests locally
=========================

From the project root, run::

  python -m pytest tests/unit
