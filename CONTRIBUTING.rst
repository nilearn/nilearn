.. _contributing:


How to help?
=============

* If you have a usage question : create a topic on `neurostars <https://neurostars.org/tag/nilearn>`_ with the nilearn tag

* You think you discovered a bug ? Create `an issue <https://github.com/nilearn/nilearn/issues/new/choose>`_
  including a minimal runnable example to showcase it (using Nilearn data) as well as your OS and Nilearn version.

* If you have an idea for a new feature, check if it is in the scope of the project
  and feel free to open an issue to discuss it.

* Want to contribute code ? Thank you very much!

  * For new features, please be sure to create an issue first, to discuss
    whether it can be included and its specifications.
  * To help with known issues, please check
    `good first issues <https://github.com/nilearn/nilearn/labels/Good%20first%20issue>`_
    to get started, `known bugs <https://github.com/nilearn/nilearn/labels/Bug>`_
    or `proposed enhancements <https://github.com/nilearn/nilearn/labels/Enhancement>`_.
  * In any case, before sending code, be sure to **follow the** `Contribution Guidelines`_.
  * The section `Setting up your environment`_ will get you ready to contribute.

* If you think documentation can be improved, you can directly send proposed
  improvements in `a pull request <https://github.com/nilearn/nilearn/pulls>`_.


How do we decide what code goes in?
====================================

Scope of the project
---------------------

Nilearn is an Open-source Python package for visualizing and analyzing human
brain MRI data.
It provides statistical and machine-learning tools for brain mapping,
connectivity estimation and predictive modelling.
It brings visualization tools with instructive documentation & open community.

Nilearn targets ease of use, but as Python code.
In other words, we will not add graphical user interfaces, but we want our
code to be as easy to understand as possible, with easy prototyping and
debugging, even for beginners in Python.

We are parsimonious in the way we add features to the project, as it
puts on weight.
To assess new features, our main concern is their usefulness to a number of
our users.
To make Nilearn high-quality and sustainable we also weigh their benefits
(i.e., new features, ease of use) with their cost (i.e., complexity of the code,
runtime of the examples). As a rule of thumb:

* To be accepted, new features must be **in the scope of the project** and
  correspond to an **established practice** (typically as used in scientific
  publications).

* It must have a concrete use case, illustrated with a **simple example** in the
  Nilearn documentation to teach it easily to end-users.

* It must be **thoroughly tested**, and respect **coding conventions** of the
  existing codebase.

* Features introducing new dependencies will generally not be accepted.

* Downloaders for new atlases are welcome if they comes with an example.

* Downloaders for new datasets are usually discouraged. We will consider adding
  fetchers only for light datasets which are needed to demo and teach features.

Exhaustive criteria used in the review process are detailed in the **contribution
guide below**.
Be sure to read and follow them so that your code can be accepted quickly.


Who makes decisions
--------------------

We strongly aim to be a community oriented project where decisions are
made based on consensus according to the criteria described above.
Discussions are public, held on issues and pull requests in Github.
All modifications of the codebase are ultimately checked during a reviewing
process, where maintainers or contributors make sure they respect the
:ref:`contribution_guidelines`.
To be merged, a pull request usually needs to be accepted by two maintainers.
In case a consensus does not emerge easily, the decisions are made by the
core contributors, i.e., people with write access to the repository, as
listed :ref:`here <core_devs>`.

How to contribute to nilearn
=============================

This project, hosted on https://github.com/nilearn/nilearn, is a community
effort, and everyone is welcome to contribute.
We value very much your feedback and opinion on features that should be
improved or added.
All discussions are public and held on relevant issues or pull requests.
To discuss your matter, please comment on a relevant
`issue <https://github.com/nilearn/nilearn/issues>`_ or open a new one.

The best way to contribute and to help the project is to start working on known
issues such as `good first issues <https://github.com/nilearn/nilearn/labels/Good%20first%20issue>`_,
`known bugs <https://github.com/nilearn/nilearn/labels/Bug>`_ or
`proposed enhancements <https://github.com/nilearn/nilearn/labels/Enhancement>`_.
If an issue does not already exist for a potential contribution, we ask that
you first open an `issue <https://github.com/nilearn/nilearn/issues>`_ before
sending a :ref:`pull request` to discuss scope and potential design choices
in advance.

.. _contribution_guidelines:

Contribution Guidelines
------------------------

When modifying the codebase, we ask every contributor to respect common
guidelines.
Those are inspired from `scikit-learn
<https://scikit-learn.org/stable/developers/contributing.html#contributing-code>`_
and ensure Nilearn remains simple to understand, efficient and maintainable.
For example, code needs to be tested and those tests need to run quickly in order
not to burden the development process.
To keep continuous integration efficient with our limited infrastructure,
running all the examples must lead to downloading a limited amount of data
(gigabytes) and execute in a reasonable amount of time (less than an hour).
Those guidelines will hence be enforced during the reviewing process.
The section `Setting up your environment`_ will help you to quickly get familiar
with the tools we use for development and deployment.

+--------------------+-------------+-----------------------------------------------------+
|                    | Which PR ?  |        Guidelines                                   |
+====================+=============+=====================================================+
|                    |             | - Clear name                                        |
|                    |             | - Link issue through mention :"Closes #XXXX"        |
|  `PR Structure`_   |    Any      | - Clearly outline goals and changes proposed        |
|                    |             | - Doesn't include "unrelated" code change           |
|                    |             | - Add entry in "doc/whats_new.rst"                  |
+--------------------+-------------+-----------------------------------------------------+
|                    |             | - Variables, functions, arguments have clear names  |
|                    |             | - Easy to read, PEP8_                               |
|   `Coding Style`_  |    Any      | - Public functions have docstring (numpydoc_ format)|
|                    |             | - Low redundancy                                    |
|                    |             | - No new dependency                                 |
|                    |             | - Backward compatibility                            |
+--------------------+-------------+-----------------------------------------------------+
|                    |             | - Test type is adapted to function behavior         |
|                    |             | - Tests pass continuous integration                 |
|                    |  Bugfixes   | - Coverage doesn't decrease                         |
|      `Tests`_      | New features| - Fast, using small mocked data                     |
|                    |             | - Atomic (one per function) and seeded              |
|                    |             | - For Bugfixes: non-regression test                 |
+--------------------+-------------+-----------------------------------------------------+
|                    |             | - Clearly showcase benefits                         |
|      Examples      | New features| - Run in a few seconds                              |
|                    |             | - Use light data (generated or from Nilearn)        |
|                    |             | - Renders well after build                          |
+--------------------+-------------+-----------------------------------------------------+
|                    |             | - Simple and didactic                               |
|  `Documentation`_  |    Any      | - Links to relevant examples                        |
|                    |             | - Renders well after build                          |
|                    |             | - Doesn't include code                              |
+--------------------+-------------+-----------------------------------------------------+

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html

PR Structure
-------------

A new pull request must have a clear scope, conveyed through its name, a
reference to the issue it targets (through the exact mention "Closes #XXXX"),
and a synthetic summary of its goals and main steps.
When working on big contributions, we advise contributors to split them into
several PRs when possible.
This has the benefit of making code changes clearer, making PRs easier to review,
and overall smoothening the whole process.
No changes unrelated to the PR should be included.

When relevant, PR names should also include tags if they fall in various
categories.
When opening a PR, the authors should include the [WIP] tag in its name, or use
github draft mode. When ready for review, they should switch the tag to [MRG] or
can switch it back to normal mode.
Other tags can describe the PR content : [FIX] for a bugfix, [DOC] for a
change in documentation or examples, [ENH] for a new feature and [MAINT] for
maintenance changes.

Coding Style
-------------

Nilearn codebase follow PEP8_ styling.
The main conventions we enforce are : line length < 80, spaces around operators,
meaningful variable names, function names are underscore separated
(e.g., ``a_nice_function``) and as short as possible,
public functions exposed in their parent module's init file,
private function names preceded with a "_" and very explicit,
classes in CamelCase, 2 empty lines between functions or classes.
Each function and class must come with a “docstring” at the top of the function
code, using numpydoc_ formatting.
They must summarize what the function does and document every parameter.


Tests
------
When fixing a bug, the first step is to write a minimal test that fails because
of it, and then write the bugfix to make this test pass.
For new code you should have roughly one test_function per function covering
every line and testing the logic of the function.
They should run on small mocked data, cover a representative range of parameters.

Tests must be seeded to avoid random failures.
For objects using random seeds (e.g. scikit-learn estimators), pass either
a  `np.random.RandomState` or an `int` as the seed.
When your test use random numbers,  those must be generated through:

.. code-block:: python

    rng = np.random.RandomState(0)
    my_number = rng.normal()

To check your changes worked and didn't break anything run `pytest nilearn`.
To do quicker checks it's possible to run only a subset of tests::

      pytest -v test_module.py


Documentation
---------------

Documentation must be understandable by people from different backgrounds.
The “narrative” documentation should be an introduction to the concepts of
the library.
It includes very little code and should first help the user figure out which
parts of the library he needs and then how to use it.
It must be full of links, of easily-understandable titles, colorful boxes and
figures.

Examples take a hands-on approach focused on a generic usecase from which users
will be able to adapt code to solve their own problems.
They include plain text for explanations, python code and its output and
most importantly figures to depict its results.
Each example should take only a few seconds to run.

To build our documentation, we are using
`sphinx <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_ for the
main documentation and
`sphinx-gallery <https://sphinx-gallery.github.io/stable/index.html>`_ for the
example tutorials. If you want to work on those, check out next section to
learn how to use those tools to build documentation.

.. _git_repo:

Setting up your environment
============================

Installing
----------

Here are the key steps you need to go through to copy the repo before contributing:

1. fork the repo from github (fork button in the top right corner of our `main github page <https://github.com/nilearn/nilearn>`_) and clone your fork locally::

      git clone git@github.com:<your_username>/nilearn.git

2. (optional but highly recommended) set up a conda environment to work on and activate it::

      conda create -n nilearn
      conda activate nilearn

3. install the forked version of `nilearn`::

      pip install . -e

4. install development dependencies::

      pip install -r requirements-dev.txt

5. check that all tests pass with (this can take a while)::

      pytest nilearn


Contributing
------------

Here are the key steps you need to go through to contribute code to `nilearn`:

1. open or join an already existing issue explaining what you want to work on

2. on your fork, create a new branch from main::

      git checkout -b your_branch

3. implement and commit your changes on this branch (don't forget to write tests!)

4. run the tests locally (to go faster, only run tests which are relevant to what
   you work on with, for example)::

      pytest -v nilearn/plotting/tests/test_surf_plotting.py

5. push your changes to your online fork::

      git push

6. in github, open a pull request from your online fork to the main repo
   (most likely from `your_fork:your_branch` to `nilearn:main`).

7. check that all continuous integration tests pass

For more details about the Fork Clone Push workflows, read `here <https://guides.github.com/activities/forking/>`_.


Building documentation
----------------------

If you wish to build documentation:

1. First, ensure that you have installed sphinx and sphinx-gallery. When in your
   fork top folder, you can install the required packages using::

      pip install -r requirements-build-docs.txt

2. Then go to ``nilearn/examples`` or ``nilearn/doc`` and make needed changes
   using `reStructuredText files <https://www.sphinx-doc.org/en/2.0/usage/restructuredtext/basics.html>`_

3. You can now go to `nilearn/doc` and build the examples locally::

      make html-strict

   or, if you do not have make install (for instance under Windows)::

      python3 -m sphinx -b html -d _build/doctrees . _build/html

   if you don't need the plots, a quicker option is::

      make html-noplot

4. Visually review the output in ``nilearn/doc/_build/html/auto_examples/``.
   If all looks well and there were no errors, commit and push the changes.

5. You can now open a Pull Request from Nilearn's Pull Request page.

6. Request the CI builds the full documentation from your branch::

      git commit --allow-empty -m "[circle full] request full build"

.. tip::
    When generating documentation locally, you can build only specific files
    to reduce building time. To do so, use the ``filename_pattern``::

       python3 -m sphinx -D sphinx_gallery_conf.filename_pattern=plot_decoding_tutorial.py -b html -d _build/doctrees . _build/html


Additional cases
=================

How to contribute an atlas
---------------------------

We want atlases in nilearn to be internally consistent. Specifically,
your atlas object should have three attributes (as with the existing
atlases):

- ``description`` (bytes): A text description of the atlas. This should be
  brief but thorough, describing the source (paper), relevant information
  related to its construction (modality, dataset, method), and, if there is
  more than one map, a description of each map.
- ``labels`` (list): a list of string labels corresponding to each atlas
  label, in the same (numerical) order as the atlas labels
- ``maps`` (list or string): the path to the nifti image, or a list of paths

In addition, the atlas will need to be called by a fetcher. For example, see `here <https://github.com/nilearn/nilearn/blob/main/nilearn/datasets/atlas.py>`__.

Finally, as with other features, please provide a test for your atlas.
Examples can be found `here
<https://github.com/nilearn/nilearn/blob/main/nilearn/datasets/tests/test_atlas.py>`__


How to contribute a dataset fetcher
------------------------------------

The :mod:`nilearn.datasets` module provides functions to download some
neuroimaging datasets, such as :func:`nilearn.datasets.fetch_haxby` or
:func:`nilearn.datasets.fetch_atlas_harvard_oxford`. The goal is not to provide a comprehensive
collection of downloaders for the most widely used datasets, and this would be
outside the scope of this project. Rather, this module provides data downloading utilities that are
required to showcase nilearn features in the example gallery.

Downloading data takes time and large datasets slow down the build of the
example gallery. Moreover, downloads can fail for reasons we do not control,
such as a web service that is temporarily unavailable. This is frustrating for
users and a major issue for continuous integration (new code cannot be merged
unless the examples run successfully on the CI infrastructure). Finally,
datasets or the APIs that provide them sometimes change, in which case the
downloader needs to be adapted.

As for any contributed feature, before starting working on a new downloader,
we recommend opening an issue to discuss whether it is necessary or if existing
downloaders could be used instead.


To add a new fetcher, ``nilearn.datasets.utils`` provides some helper functions,
such as ``_get_dataset_dir`` to find a directory where the dataset is or will be
stored according to the user's configuration, or ``_fetch_files`` to load files
from the disk or download them if they are missing.

The new fetcher, as any other function, also needs to be tested (in the relevant
submodule of ``nilearn.datasets.tests``). When the tests run, the fetcher does
not have access to the network and will not actually download files. This is to
avoid spurious failures due to unavailable network or servers, and to avoid
slowing down the tests with long downloads.
The functions from the standard library and the ``requests`` library that
nilearn uses to download files are mocked: they are replaced with dummy
functions that return fake data.

Exactly what fake data is returned can be configured through the object
returned by the ``request_mocker`` pytest fixture, defined in
``nilearn.datasets._testing``. The docstrings of this module and the ``Sender``
class it contains provide information on how to write a test using this fixture.
Existing tests can also serve as examples.

Maintenance
=================

More information about the project organization, conventions, and maintenance
process can be found there : :ref:`maintenance_process`.
