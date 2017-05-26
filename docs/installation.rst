.. highlight:: shell

Installation
============

To install ELFI, run this command in your terminal:

.. code-block:: console

    pip install elfi

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


ELFI is currently tested only with Python 3.5. If you are new to Python, perhaps the simplest way to install it is Anaconda_

.. _Anaconda: https://www.continuum.io/downloads

Optional dependencies
---------------------

Optionally you may wish to install also the following packages:

* `graphviz` for drawing graphical models (Graphviz_ must be installed separately)

.. _Graphviz: http://www.graphviz.org

Virtual environment using Anaconda
----------------------------------

If you want to create a virtual environment before installing, you can do so with Anaconda:

.. code-block:: console

    conda create -n elfi python=3.5 scipy
    source activate elfi
    pip install elfi


Potential problems with installation
------------------------------------

ELFI depends on several other Python packages, which have their own dependencies. Resolving these may sometimes go wrong:

* If you receive an error about missing `numpy`, please install it first.
* If you receive an error about `yaml.load`, install `pyyaml`.
* On OS X with Anaconda virtual environment say `conda install python.app` and then use `pythonw` instead of `python`.
* Note that ELFI currently supports Python 3.5 only, although 3.x may work as well.

From sources
------------

The sources for ELFI can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone https://github.com/elfi-dev/elfi.git

Or download the `tarball`_:

.. code-block:: console

    curl  -OL https://github.com/elfi-dev/elfi/tarball/master

Note that for development it is recommended to base your work on the `dev` branch instead of `master`.

Once you have a copy of the source, you can install it with:

.. code-block:: console

   pip install -e .

This will install ELFI along with its default requirements.

.. _Github repo: https://github.com/elfi-dev/elfi
.. _tarball: https://github.com/elfi-dev/elfi/tarball/master

