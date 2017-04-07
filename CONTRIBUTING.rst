.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/elfi-dev/elfi/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

ELFI could always use more documentation, whether as part of the
official ELFI docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/elfi-dev/elfi/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `ELFI` for local development.

1. Fork the `elfi` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/elfi.git

3. Install your local copy and the development requirements into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development. Due to a bug in the pip installation of GPy numpy needs to be installed manually.::

    $ mkvirtualenv elfi
    $ cd elfi/
    $ pip install numpy
    $ make dev

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.
   
5. Follow the `Style Guidelines`_

6. When you're done making changes, check that your changes pass flake8 and the tests::

    $ make lint
    $ make test

  Also make sure that the docstrings of your code are formatted properly::

    $ make docs

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Style Guidelines
----------------

The projects follows the `Khan Academy Style Guide <https://github.com/Khan/style-guides/blob/master/style/python.md>`_. Except that we use numpy style docstrings instead of Google style docstrings.
   
See `this example <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_ for how to format the docstrings.

Additional Style Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use the ``.format()`` string method instead of the old percent operator. For more information see `PyFormat <https://pyformat.info/>`_.
- Use the type hinting syntax suggested `here <https://www.jetbrains.com/help/pycharm/2016.1/type-hinting-in-pycharm.html>`_ in the docstrings.
  
Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 2.7,  3.5 and later. Check
   https://travis-ci.org/elfi-dev/elfi/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ py.test tests.test_elfi

