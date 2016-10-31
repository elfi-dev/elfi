ELFI
====

.. image:: https://img.shields.io/pypi/v/elfi.svg
        :target: https://pypi.python.org/pypi/elfi

.. image:: https://img.shields.io/travis/HIIT/elfi.svg
        :target: https://travis-ci.org/HIIT/elfi

.. image:: https://readthedocs.org/projects/elfi/badge/?version=latest
        :target: https://elfi.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
 
 
Engine for likelihood-free inference


..
   Installation
   -------------
   ::

     pip install elfi

Developer installation
~~~~~~~~~~~~~~~~~~~~~~~
::

  git clone https://github.com/HIIT/elfi.git
  cd elfi
  pip install numpy
  pip install -r requirements-dev.txt
  pip install -e .
  
It is recommended to create a virtual environment for development before installing.

Virtual environment using Anaconda
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Below an example how to create a virtual environment named ``elfi`` using Anaconda::

    conda create -n elfi python=3* scipy

Then activate it::

    source activate elfi
