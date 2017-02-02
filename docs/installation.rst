.. highlight:: shell

============
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

Currently it is required to use Distributed 1.14.3.

Optional dependencies:

- `graphviz` for drawing graphical models (needs `dot` from the full Graphviz_)
- `unqlite` for using NoSQL storage

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

    git clone https://github.com/HIIT/elfi.git

Or download the `tarball`_:

.. code-block:: console

    curl  -OL https://github.com/HIIT/elfi/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    python setup.py install

.. _Github repo: https://github.com/HIIT/elfi
.. _tarball: https://github.com/HIIT/elfi/tarball/master
