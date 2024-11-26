.. _contributing:


Code of Conduct
===============

By participating in Nilearn, you agree to abide by the :nipy:`NIPY Code of Conduct <conduct.html>`

How to get help?
================

If you have issues when using Nilearn, or if you have questions on how to use it, please don't hesitate to reach out!

There are currently three ways to interact with the Nilearn team: through the :neurostars:`neurostars <>` forum, our :nilearn-gh:`github <>` issues, and through our weekly `drop-in hours <https://tinyurl.com/nilearn-drop-in-hour>`_, usually **every Wednesday from 4pm to 5pm UTC**.
We post on our `X account <https://twitter.com/nilearn>`_ in advance to let you know if the drop-in hours are happening that week.

If you have a *usage question*, that is if you need help troubleshooting scripts using Nilearn, we would appreciate it if you either ask it during the drop-in hours or create a topic on :neurostars:`neurostars <>` with the "nilearn" tag.
Asking questions or reporting issues is always valuable because it will help other users having the same problem. So, please don't hold onto a burning question!

We ask that you *don't* open an issue on :nilearn-gh:`GitHub <>` for usage questions. We use our :nilearn-gh:`GitHub <>` issue board for bug reports, feature requests, and documentation changes only.

How to help the project?
========================

If you are interested in contributing to the Nilearn project, we thank you very much. Note that there are multiple ways to help us, and not all of them require writing code.

Report bugs or discuss enhancement ideas
----------------------------------------

We welcome open discussion around improvements---both to the documentation as well as to the code base---through our GitHub issue board!

* If you think you have discovered a bug,
  please start by searching through the existing :nilearn-gh:`issues <issues>`
  to make sure it has not already been reported. If the bug has not been reported yet,
  create an :nilearn-gh:`new issue <issues/new/choose>`
  including a `minimal runnable example <https://stackoverflow.com/help/minimal-reproducible-example>`_
  to showcase it (using :ref:`nilearn.datasets <datasets_ref>`) as well as your OS and Nilearn version.

* If you have an idea for a new feature, check if it is in the :ref:`nilearn_scope`
  and feel free to open a :nilearn-gh:`new issue <issues/new/choose>` to discuss it.

* If you think the documentation can be improved, please open a :nilearn-gh:`new issue <issues/new/choose>`
  to discuss what you would like to change! This helps to confirm
  that your proposed improvements don't overlap with any ongoing work.

Answer questions
----------------

Another way to help the project is to answer questions on :neurostars:`neurostars <>`, or comment on github :nilearn-gh:`issues <issues>`.
Some :nilearn-gh:`issues <issues>` are used to gather user opinions on various questions, and any input from the community is valuable to us.

Review Pull Requests
--------------------

Any addition to the Nilearn's code base has to be reviewed and approved by several people including at least two :ref:`core_devs`.
This can put a heavy burden on :ref:`core_devs` when a lot of
:nilearn-gh:`pull requests <pulls>` are opened at the same time.
We welcome help in reviewing :nilearn-gh:`pull requests <pulls>` from any
community member.
We do not expect community members to be experts in all changes included in
:nilearn-gh:`pull requests <pulls>`, and we encourage you to concentrate on those code changes that you feel comfortable with.
As always, more eyes on a code change means that the code is more likely to work in a wide variety of contexts!

Contribute code
---------------

If you want to contribute code:

* For new features, please be sure to create a :nilearn-gh:`new issue <issues/new/choose>` first,
  to discuss whether it can be included and its specifications.

* To help with known :nilearn-gh:`issues <issues>`,
  please check :nilearn-gh:`good first issues <labels/Good%20first%20issue>`
  to get started, :nilearn-gh:`known bugs <labels/Bug>`,
  or :nilearn-gh:`proposed enhancements <labels/Enhancement>`.

Please see the :ref:`contributing_code` section for more detailed information, including
instructions for  `Setting up your environment`_ and a description of the `Contribution Guidelines`_.

How do we decide what code goes in?
====================================

The following sections explain the :ref:`nilearn_scope` and :ref:`nilearn_governance`, which jointly determine whether potential contributions will be accepted into the project.

.. _nilearn_scope:

Scope of the project
--------------------

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
runtime of the examples).

Exhaustive criteria used in the review process
are detailed in the **contribution guide below**.
Be sure to read and follow them so that your code can be accepted quickly.

As a rule of thumb:

* To be accepted, new features must be **in the scope of the project** and
  correspond to an **established practice** (typically as used in scientific
  publications).

* It must have a concrete use case, illustrated with a **simple example** in the
  Nilearn documentation to teach it easily to end-users.

* It must be **thoroughly tested**, and respect **coding conventions** of the
  existing codebase.

* Features introducing new dependencies will generally not be accepted.

Adding atlases and datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Downloaders for new atlases datasets are usually discouraged.

There is no intention to provide a comprehensive collection of downloaders
for the most widely used atlases and datasets.
**This would be outside the scope of this project.**
We will consider adding fetchers only for atlases and light datasets
which are needed to demo and teach features.

.. warning::

      Issues requesting to add a new atlas or dataset that are not linked
      to the development of a new feature or example
      will be closed as being **out of scope**.

The following projects are dedicated to host atlases and accept contributions:

* `templateflow <https://www.templateflow.org>`_
* `neuromaps <https://netneurolab.github.io/neuromaps>`_
* `neuroparc <https://github.com/neurodata/neuroparc>`_


.. _nilearn_governance:

Who makes decisions
-------------------

We strongly aim to be a community oriented project where decisions are
made based on consensus according to the criteria described above.
Discussions are public, held on :nilearn-gh:`issues <issues>` and
:nilearn-gh:`pull requests <pulls>` in Github.
All modifications of the codebase are ultimately checked during a reviewing
process, where maintainers or contributors make sure they respect the
:ref:`contribution_guidelines`.
To be merged, a pull request usually needs to be accepted by two maintainers.
In case a consensus does not emerge easily, the decisions are made by the
:ref:`core_devs`, i.e., people with write access to the repository, as
listed :ref:`here <core_devs>`.

.. _contributing_code:

How to contribute to nilearn
=============================

This project, hosted on :nilearn-gh:`\ `, is a community
effort, and everyone is welcome to contribute.
We value very much your feedback and opinion on features that should be
improved or added.
All discussions are public and held on relevant :nilearn-gh:`issues <issues>` or
:nilearn-gh:`pull requests <pulls>`.
To discuss your matter, please comment on a relevant
:nilearn-gh:`issue <issues>` or open a new one.

The best way to contribute and to help the project is to start working on known
:nilearn-gh:`issues <issues>` such as
:nilearn-gh:`good first issues <labels/Good%20first%20issue>`,
:nilearn-gh:`known bugs <labels/Bug>` or
:nilearn-gh:`proposed enhancements <labels/Enhancement>`.
If an issue does not already exist for a potential contribution, we ask that
you first open a :nilearn-gh:`new issue <issues/new/choose>` before sending a
:ref:`pull request` to discuss scope and potential design choices in advance.

.. _contribution_guidelines:

Contribution Guidelines
-----------------------

When modifying the codebase, we ask every contributor to respect common
guidelines.
Those are inspired from :sklearn:`scikit-learn <developers/contributing.html#contributing-code>`
and ensure Nilearn remains simple to understand, efficient and maintainable.
For example, code needs to be tested and those tests need to run quickly in order
not to burden the development process.
To keep continuous integration efficient with our limited infrastructure,
running all the examples must lead to downloading a limited amount of data
(gigabytes) and execute in a reasonable amount of time (less than an hour).
Those guidelines will hence be enforced during the reviewing process.
The section `Setting up your environment`_ will help you to quickly get familiar
with the tools we use for development and deployment.

+--------------------+---------------+-----------------------------------------------------+
|                    | Which PR ?    |        Guidelines                                   |
+====================+===============+=====================================================+
|                    |               | - Clear name                                        |
|                    |               | - Link issue through mention :"Closes #XXXX"        |
|  `PR Structure`_   |    Any        | - Clearly outline goals and changes proposed        |
|                    |               | - Doesn't include "unrelated" code change           |
|                    |               | - Add entry in "doc/changes/latest.rst"             |
+--------------------+---------------+-----------------------------------------------------+
|                    |               | - Variables, functions, arguments have clear names  |
|                    |               | - Easy to read, PEP8_ compliant                     |
|                    |               | - Code formatted with ruff_                         |
|                    |               | - Public functions have docstring (numpydoc_ format)|
|                    |               | - Low redundancy                                    |
|   `Coding Style`_  |    Any        | - No new dependency                                 |
|                    |               | - Backward compatibility                            |
|                    |               | - All internal imports are absolute, not relative   |
|                    |               | - Impacted docstrings have versionadded and/or      |
|                    |               |   versionchanged directives as needed.              |
|                    |               |   These should use the current dev version.         |
+--------------------+---------------+-----------------------------------------------------+
|                    |               | - Test type is adapted to function behavior         |
|                    |               | - Tests pass continuous integration                 |
|                    | - Bugfixes    | - Coverage doesn't decrease                         |
|      `Tests`_      | - New features| - Fast, using small mocked data                     |
|                    |               | - Atomic (one per function) and seeded              |
|                    |               | - For Bugfixes: non-regression test                 |
+--------------------+---------------+-----------------------------------------------------+
|                    |               | - Clearly showcase benefits                         |
|      Examples      | New features  | - Run in a few seconds                              |
|                    |               | - Use light data (generated or from Nilearn)        |
|                    |               | - Renders well after build                          |
+--------------------+---------------+-----------------------------------------------------+
|                    |               | - Simple and didactic                               |
|  `Documentation`_  |    Any        | - Links to relevant examples                        |
|                    |               | - Renders well after build                          |
|                    |               | - Doesn't include code                              |
+--------------------+---------------+-----------------------------------------------------+

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _ruff: https://docs.astral.sh/ruff/

PR Structure
------------

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

.. _changelog:

Changelog
---------

Changelog entries in ``nilearn/doc/changes/latest.rst`` should adhere to the following conventions:

- Entry in the appropriate category
- Single line per entry
- Add a "badge" corresponding to the change type (see below)
- Finish with a link to the PR and the author's profile
- New contributors to add their details to the ``authors`` section of the ``CITATION.cff`` (see below)

List of badges:

.. code-block:: rst

      :bdg-primary:`Doc`
      :bdg-secondary:`Maint`
      :bdg-success:`API`
      :bdg-info:`Plotting`
      :bdg-warning:`Test`
      :bdg-danger:`Deprecation`
      :bdg-dark:`Code`

Example entry in ``nilearn/doc/changes/latest.rst``:

.. code-block:: rst

    - :bdg-dark:`Code` Fix off-by-one error when setting ticks in :func:`~plotting.plot_surf` (:gh:`3105` by `Dimitri Papadopoulos Orfanos`_).

Associated entry in ``CITATION.cff``:

.. code-block:: yaml

      authors:

        - given-names: Dimitri Papadopoulos
          family-names: Orfanos
          website: https://github.com/DimitriPapadopoulos
          affiliation: NeuroSpin, C.E.A., Université Paris-Saclay, Gif-sur-Yvette, France
          orcid: https://orcid.org/0000-0002-1242-8990


Coding Style
------------

The nilearn codebase follows PEP8_ styling.
The main conventions we enforce are :

- line length < 80
- spaces around operators
- meaningful variable names
- function names are underscore separated (e.g., ``a_nice_function``) and as short as possible
- public functions exposed in their parent module's init file
- private function names preceded with a "_" and very explicit, see also :ref:`private_functions`
- classes in CamelCase
- 2 empty lines between functions or classes

You can check that any code you may have edited follows these conventions
by running `ruff <https://docs.astral.sh/ruff/>`__.

Documentation style
^^^^^^^^^^^^^^^^^^^

Each function and class must come with a “docstring” at the top of the function code,
using numpydoc_ formatting.
The docstring must summarize what the function does and document every parameter.

If an argument takes in a default value, it should be described
with the type definition of that argument.

See the examples below:

.. code-block:: python

      def good(x, y=1, z=None):
          """Show how parameters are documented.

          Parameters
          ----------
          x : :obj:`int`
                X

          y : :obj:`int`, default=1
                Note that "default=1" is preferred to "Defaults to 1".

          z : :obj:`str`, default=None

          """


      def bad(x, y=1, z=None):
          """Show how parameters should not be documented.

          Parameters
          ----------
          x :
                The type of X is not described

          y : :obj:`int`
                The default value of y is not described.

          z : :obj:`str`
                Defaults=None.
                The default value should be described after the type.
          """

Additionally, we consider it best practice to write modular functions;
i.e., functions should preferably be relatively short and do *one* thing.
This is also useful for writing unit tests.

Writing small functions is not always possible, and we do not recommend trying to reorganize larger,
but well-tested, older functions in the codebase, unless there is a strong reason to do so (e.g., when adding a new feature).

APIs of nilearn objects
^^^^^^^^^^^^^^^^^^^^^^^

Estimated Attributes
""""""""""""""""""""

Attributes that have been estimated from the data
should always have a name ending with trailing underscore.
For example the coefficients of some regression estimator
would be stored in a ``coef_`` attribute after ``fit`` has been called.

The estimated attributes are expected to be overridden when you call ``fit`` a second time.

This follows the `scikit-learn convention <https://scikit-learn.org/stable/developers/develop.html#estimated-attributes>`_.

.. _private_functions:

Guidelines for Private Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We start a name with a leading underscore to indicate that it is an internal implementation detail,
not to be accessed directly from outside, of the enclosing context:

- the parent module (for a submodule name),
- or the module (for the name of a top-level function, class or global variable),
- or the class (for a method or attribute name).

Moreover, modules explicitly declare their interface through the ``__all__`` attribute,
and any name not listed in ``__all__`` should not be accessed from outside the module

In some cases when private functions are in a private module (filename beginning with an underscore),
but are used outside of that file, we do not name them with a leading underscore.

Example:

.. code-block:: rst

      nilearn
      ├── image.py             # part of public API
      ├── __init__.py
      ├── maskers              # part of public API
      │   ├── __init__.py
      │   ├── nifti_masker.py  # part of public API
      │   └── _validation.py   # private to the maskers module
      └── _utils.py            # private to the nilearn module

Code inside ``maskers._validation.py``:

.. code-block:: python

      import numpy as np  # not part of the public API

      __all__ = ["check_mask_img", "ValidationError"]  # all symbols in the public API


      def check_mask_img(mask_img):
          """Public API of _validation module

          can be used in nifti_masker module
          but not the image module (which cannot import maskers._validation),
          unless maskers/__init__.py imports it and lists it in __all__
          to make it part of the maskers module's public API
          """

          return _check_mask_shape(mask_img) and _check_mask_values(mask_img)


      def _check_mask_shape(mask_img):
          """Private internal of _validation, cannot be used in nifti_masker"""


      def _check_mask_values(mask_img):
          """Private internal of _validation, cannot be used in nifti_masker"""


      class ValidationError(Exception):
          """Public API of _validation module"""


      class _Validator:
          """Private internal of the _validation module"""

          def validate(self, img):
              """Public API of _Validator"""

          def _validate_shape(self, img):
              """Private internal of the _Validator class.

              As we don't use the double leading underscore in nilearn we
              cannot infer from the name alone if it is considered to be
              exposed to subclasses or not.

              """

..
      Source: Jerome Dockes https://github.com/nilearn/nilearn/issues/3628#issuecomment-1515211711

Guidelines for HTML and CSS
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use `prettier <https://prettier.io/>`_ to format HTML and CSS.

This is implemented via a pre-commit hook (see below)
that can be run with

.. code-block:: bash

      pre-commit run --all-files prettier

Pre-commit
----------

We use `pre-commit <https://pre-commit.com/>`__
to run a set of linters and autoformatters on the codebase.

To install pre-commit, run:

.. code-block:: bash

      pip install pre-commit

Then run the following to install the pre-commit hooks:

.. code-block:: bash

      pre-commit install

Pre-commit will then run all those hooks on the files you have staged for commit.
Note that if some of those hooks fail you may have to edit some files and stage them again.

Tests
-----

When fixing a bug, the first step is to write a minimal test that fails because
of it, and then write the bugfix to make this test pass.
For new code you should have roughly one test function per function covering
every line and testing the logic of the function.
They should run on small mocked data, cover a representative range of parameters.

.. hint::

      It is easier to write good unit tests for short, self-contained functions.
      Try to keep this in mind when you write new functions.
      For more information about this coding approach,
      see `test-driven development <https://en.wikipedia.org/wiki/Test-driven_development>`_.

We use `pytest <https://docs.pytest.org/en/6.2.x/contents.html>`_ to run our tests.

If you are not familiar with pytest,
have a look at this `introductory video <https://www.youtube.com/watch?v=mzlH8lp4ISA>`_
by one of the pytest core developer.

In general tests for a specific module (say ``nilearn/image/image.py``)
are kept in a ``tests`` folder in a separate module
with a name that matches the module being tested
(so in this case ``nilearn/image/tests/test_image.py``).

When you have added a test you can check that your changes worked
and didn't break anything by running ``pytest nilearn``.
To do quicker checks it's possible to run only a subset of tests:

.. code-block:: bash

      pytest -v nilearn/module/tests/test_module.py

Fixtures
^^^^^^^^

If you need to do some special "set up" for your tests
(for example you need to generate some data, or a NiftiImage object or a file...)
you can use `pytest fixtures <https://docs.pytest.org/en/6.2.x/fixture.html>`_
to help you mock this data
(more information on pytest fixtures in `this video <https://www.youtube.com/watch?v=ScEQRKwUePI>`_).

Fixture are recognizable because they have a ``@pytest.fixture`` decorator.
Fixtures that are shared by many tests modules can be found in ``nilearn/conftest.py``
but some fixures specific to certain modules can also be kept in that testing module.

Before adding new fixtures, first check those that exist
in the test modules you are working in or in ``nilearn/conftest.py``.

Seeding
^^^^^^^

Many tests must be seeded to avoid random failures.
When your test use random numbers,
you can seed a random number generator with ``numpy.random.default_rng``
like in the following examples:

.. code-block:: python

      def test_something():
          # set up
          rng = np.random.default_rng(0)
          my_number = rng.normal()

          # the rest of the test

You can also use the ``rng`` fixture.

.. code-block:: python

      def test_something(rng):
          # set up
          my_number = rng.normal()

          # the rest of the test

Documentation
-------------

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


Continuous integration
----------------------

Please note that if one of the following markers appear in the latest commit message, the following actions are taken.

============================ ===================
Commit Message Marker        Action Taken by CI
============================ ===================
[skip ci]                    Gtihub CI is skipped completely. Several other options are also possible, see `github documentation <https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-workflow-runs/skipping-workflow-runs>`_).
[skip test]                  Skip running the tests.
[skip doc]                   Skip building the doc.
[test nightly]               Run tests on the nightly build of Nilearn's dependencies.
[full doc]                   Run a full build of the documentation (meaning that all examples will be run).
[example] name_of_example.py Run partial documentation build but will run the requested example.
[force download]             Force a download of all the dataset required for the build of the documentation.
============================ ===================

Setting up your environment
===========================

Installing
----------

Here are the key steps you need to go through to copy the repo before contributing:

1. fork the repo from github (fork button in the top right corner of our :nilearn-gh:`main github page <>`)
   and clone your fork locally:

.. code-block:: bash

      git clone git@github.com:<your_username>/nilearn.git

2. (optional but highly recommended) set up a virtual environment to wor
   in using whichever environment management tool you're used to and activate it.
   For example:

.. code-block:: bash

      python3 -m venv nilearn
      source nilearn/bin/activate

or:

.. code-block:: bash

      conda create -n nilearn pip
      conda activate nilearn

3. install the forked version of ``nilearn``

.. admonition:: Recommendation

      When you install nilearn, it will use the version stored in the version control system
      (in this case git)
      to get the version number you would see if you typed in your terminal:

      .. code-block:: bash

            pip show nilearn

      or:

      .. code-block:: bash

            python -c "import nilearn; print(nilearn.__version__)"

      To make sure that you get the correct version number, you must fetch
      all the git tags from the nilearn github repository,
      by running the following commands:

      .. code-block:: bash

            # add the nilearn repo as an "upstream" remote
            git remote add upstream https://github.com/nilearn/nilearn.git
            # fetch all the tags
            git fetch --all
            # check that you got all the tags
            git tag --list

You can then install nilearn in editable mode:

.. code-block:: bash

      pip install -e '.[dev]'

This installs your local version of Nilearn,
along with all dependencies necessary for developers (hence the ``[dev]`` tag).
For more information about the dependency installation options, see ``pyproject.toml``.
The installed version will also reflect any changes you make to your code.


4. check that all tests pass with (this can take a while):

.. code-block:: bash

      pytest nilearn

5. (optional) install `pre-commit <https://pre-commit.com/#usage>`_ hooks
   to run the linter and other checks before each commit:

.. code-block:: bash

      pre-commit install


Contributing
------------

Here are the key steps you need to go through to contribute code to ``nilearn``:

1. open or join an already existing issue explaining what you want to work on

2. on your fork, create a new branch from main:

.. code-block:: bash

      git checkout -b your_branch

3. implement changes, lint and format

.. admonition:: Recommendation

    To lint your code and verify PEP8 compliance, you can run
    `ruff <https://docs.astral.sh/ruff/>`_ locally on the
    changes you have made.

    .. code-block:: bash

        ruff check --fix <path_to_edited_file>

    To format your code, you can also use ruff and run:

    .. code-block:: bash

        ruff format <path_to_edited_file>

    Note that if you installed pre-commit and the pre-commit hooks,
    those commands will be run automatically before each commit.

4. commit your changes on this branch (don't forget to write tests!)

5. run the tests locally (to go faster, only run tests which are relevant to what
   you work on with, for example):

.. code-block:: bash

      pytest -v nilearn/plotting/tests/test_surf_plotting.py

6. push your changes to your online fork:

.. code-block:: bash

      git push

7. in github, open a pull request from your online fork to the main repo
   (most likely from ``your_fork:your_branch`` to ``nilearn:main``).

8. check that all continuous integration tests pass

For more details about the Fork Clone Push workflows, read `here <https://guides.github.com/activities/forking/>`_.


Building documentation
----------------------

If you wish to build documentation:

1. First, ensure that you have installed sphinx and sphinx-gallery. When in your
   fork top folder, you can install the required packages using:

.. code-block:: bash

      pip install '.[doc]'

2. Then go to ``nilearn/examples`` or ``nilearn/doc`` and make needed changes
   using `reStructuredText files <https://www.sphinx-doc.org/en/2.0/usage/restructuredtext/basics.html>`_

3. You can now go to ``nilearn/doc`` and build the examples locally:

.. code-block:: bash

      make html-strict

or, if you do not have make install (for instance under Windows):

.. code-block:: bash

      python3 -m sphinx -b html -d _build/doctrees . _build/html

The full build can take a very long time.
So if you don't need the plots, a quicker option is:

.. code-block:: bash

      make html-noplot

4. Visually review the output in ``nilearn/doc/_build/html/auto_examples/``.
   If all looks well and there were no errors, commit and push the changes.

5. You can now open a Pull Request from Nilearn's Pull Request page.

6. Request the CI builds the full documentation from your branch:

.. code-block:: bash

      git commit --allow-empty -m "[full doc] request full build"

.. tip::
    When generating documentation locally, you can build only specific files
    to reduce building time. To do so, use the ``filename_pattern``:

.. code-block:: bash

      python3 -m sphinx -D sphinx_gallery_conf.filename_pattern=\\
      plot_decoding_tutorial.py -b html -d _build/doctrees . _build/html


Additional cases
================

How to contribute an atlas
--------------------------

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

In addition, the atlas will need to be called by a fetcher. For example, see :nilearn-gh:`here <blob/main/nilearn/datasets/atlas.py>`.

Finally, as with other features, please provide a test for your atlas.
Examples can be found :nilearn-gh:`here <blob/main/nilearn/datasets/tests/test_atlas.py>`.


How to contribute a dataset fetcher
-----------------------------------

The :mod:`nilearn.datasets` module provides functions to download some
neuroimaging datasets, such as :func:`nilearn.datasets.fetch_haxby` or
:func:`nilearn.datasets.fetch_atlas_harvard_oxford`.

Downloading data takes time and large datasets slow down the build of the
example gallery. Moreover, downloads can fail for reasons we do not control,
such as a web service that is temporarily unavailable. This is frustrating for
users and a major issue for continuous integration (new code cannot be merged
unless the examples run successfully on the CI infrastructure). Finally,
datasets or the APIs that provide them sometimes change, in which case the
downloader needs to be adapted.

As for any contributed feature, before starting working on a new downloader,
we recommend opening a :nilearn-gh:`new issue <issues/new/choose>` to discuss
whether it is necessary or if existing downloaders could be used instead.

To add a new fetcher, ``nilearn.datasets.utils`` provides some helper functions,
such as ``get_dataset_dir`` to find a directory where the dataset is or will be
stored according to the user's configuration, or ``fetch_files`` to load files
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
``nilearn.datasets.tests._testing``. The docstrings of this module and the
``Sender`` class it contains provide information on how to write a test using
this fixture. Existing tests can also serve as examples.

Maintenance
===========

More information about the project organization, conventions, and maintenance
process can be found there : :ref:`maintenance_process`.
