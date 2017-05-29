Quickstart
==========

To use ELFI in a project with Python 3.5 or greater::

    import elfi

ELFI includes an easy to use generative modeling syntax, where the generative model is
specified as a directed acyclic graph (DAG)::


The inferencedata generation process can then be automatically parallelized from multiple
cores up to a cluster environment.


ELFI also handles seeding the random number generators and storing of the generated data
for you so that you can easily repeat or fine tune your inference.

For tutorials, please see the Jupyter Notebooks under the `notebooks directory`_. Feel
free to add your own in the zoo_.

.. _notebooks directory: https://github.com/elfi-dev/notebooks
.. _zoo: https://github.com/elfi-dev/zoo
