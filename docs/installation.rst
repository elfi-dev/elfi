.. highlight:: shell

Installation
============

ELFI requires Python 3.7 or greater (see below how to install). To install ELFI, simply
type in your terminal:

.. code-block:: console

    pip install elfi

In some OS you may have to first install ``numpy`` with ``pip install numpy``. If you don't
have `pip`_ installed, this `Python installation guide`_ can guide you through the
process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Installing Python 3.7
---------------------

If you are new to Python, perhaps the simplest way to install it is with Anaconda_ that
manages different Python versions. After installing Anaconda, you can create a Python 3.7.
environment with ELFI:

.. code-block:: console

    conda create -n elfi python=3.7 numpy
    source activate elfi
    pip install elfi

.. _Anaconda: https://www.continuum.io/downloads

Optional dependencies
---------------------

We recommend to install:

* ``graphviz`` for drawing graphical models (``pip install graphviz`` requires Graphviz_ binaries).

.. _Graphviz: http://www.graphviz.org

Potential problems with installation
------------------------------------

ELFI depends on several other Python packages, which have their own dependencies.
Resolving these may sometimes go wrong:

* If you receive an error about missing ``numpy``, please install it first.
* If you receive an error about `yaml.load`, install ``pyyaml``.
* On OS X with Anaconda virtual environment say `conda install python.app` and then use `pythonw` instead of `python`.
* Note that ELFI requires Python 3.7 or greater
* In some environments ``pip`` refers to Python 2.x, and you have to use ``pip3`` to use the Python 3.x version
* Make sure your Python installation meets the versions listed in requirements_.

.. _requirements: https://github.com/elfi-dev/elfi/blob/dev/requirements.txt

Developer installation from sources
-----------------------------------

The sources for ELFI can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone https://github.com/elfi-dev/elfi.git

Or download the development `tarball`_:

.. code-block:: console

    curl -OL https://github.com/elfi-dev/elfi/tarball/dev

Note that for development it is recommended to base your work on the `dev` branch instead
of `master`.

Once you have a copy of the source, go to the folder and type:

.. code-block:: console

   pip install -e .

This will install ELFI along with its default requirements. Note that the dot in the end
means the current folder.

.. _Github repo: https://github.com/elfi-dev/elfi
.. _tarball: https://github.com/elfi-dev/elfi/tarball/dev

Docker container
----------------

A simple Dockerfile with Jupyter support is also provided. This is especially suitable 
for running tests. Please see `Docker documentation`_ for details.

.. _Docker documentation: https://docs.docker.com/

.. code-block:: console

    git clone --depth 1 https://github.com/elfi-dev/elfi.git
    cd elfi
    make docker-build  # builds the image with requirements for dev
    make docker  # runs a container with live elfi directory

To open a Jupyter notebook, run

.. code-block:: console
    
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

within the container and then on host open the page `localhost:8888`_.

.. _localhost:8888: http://localhost:8888
