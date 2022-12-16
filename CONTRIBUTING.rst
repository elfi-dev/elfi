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

`ELFI` is a project with dozens of collaborators, so organization is key to making our contributions effective and avoid reword. Thus, in addition to the recommendations below we strongly recommend reading our `Wiki <https://github.com/elfi-dev/elfi/wiki>`_ to see what is the suggested git workflow procedure for your type of contribution.

Ready to contribute? Here's how to set up `ELFI` for local development.

1. Fork the `elfi` repo on GitHub.
2. Clone your fork locally and add the base repository as a remote::

    $ git clone git@github.com:your_github_handle_here/elfi.git
    $ cd elfi
    $ git remote add upstream git@github.com:elfi-dev/elfi.git

3. Make sure you have `Python 3 <https://www.python.org/>`_ and
   `Anaconda Distribution <https://www.anaconda.com/>`_ installed on your
   machine. Check your conda and Python versions. Currently supported Python versions
   are 3.7, 3.8, 3.9, 3.10::

   $ conda -V
   $ python -V

4. Install your local copy and the development requirements into a conda
   environment. You may need to replace "3.7" in the first line with the python
   version printed in the previous step::

    $ conda create -n elfi python=3.7 numpy
    $ source activate elfi
    $ cd elfi
    $ make dev

5. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

6. Follow the `Style Guidelines`_

7. When you're done making changes, check that your changes pass flake8 and the tests::

    $ make lint
    $ make test

   You may run ``make test-notslow`` instead of ``make test`` *as long as your proposed changes are unrelated to BOLFI*.

   Also make sure that the docstrings of your code are formatted properly::

    $ make docs

8. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."

9. After committing your changes, you may sync with the base repository if there has been changes::
    $ git fetch upstream
    $ git rebase upstream/dev

10. Push the changes::
    $ git push origin name-of-your-bugfix-or-feature

11. Submit a pull request through the GitHub website.

Style Guidelines
----------------

The Python code in ELFI mostly follows `PEP8 <http://pep8.org/>`_, which is considered the de-facto code style guide for Python. Lines should not exceed 100 characters.

Docstrings follow the `NumPy style <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests that will be run automatically using
   Github Actions.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.7 and later. Check
   https://github.com/elfi-dev/elfi/actions/workflows/pytest.yml
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ py.test tests.test_elfi

