.. _contributing:

Too long, didn't read
=====================

Installing
----------

Here are the key steps you need to go through to copy the repo before contributing:

1. fork the repo from github (fork button in the top right corner of our `main github page <https://github.com/nilearn/nilearn>`_) and clone your fork locally: `git clone git@github.com:<your_username>/nilearn.git`
1. (optional but highly recommended) setup a conda environment to work on: `conda create -n nilearn python=3.9`
1. (optional but highly recommended) activate this environment with `conda activate nilearn`
1. install the forked version of `nilearn`: `pip install .`
1. install dependencies with `pip install -r requirements-dev.txt`
1. check that all tests pass with `pytest nilearn` (this can take a while)

Contributing
------------

Here are the key steps you need to go through to contribute code to `nilearn`:

1. open or join an already existing issue explaining what you want to work on
1. on your fork, create a new branch from master: `git checkout -b your_branch`
1. implement and commit your changes on this branch (don't forget to write tests!)
1. run the tests locally (to go faster, only run tests which are relevant to what you work on with, ex: `pytest -v nilearn/plotting/tests/test_surf_plotting.py`)
1. push your changes to your online fork `git push`
1. in github, open a pull request from your online fork to the main repo (most likely from `your_fork:your_branch` to `nilearn:master`). Make sure you mention the original issue in your pull request
1. check that all online tests pass and assign someone for a code review

How to help?
=============

* If you have a usage question : create a question on `neurostars <https://neurostars.org/>`_ with tag nilearn

* You think you discovered a bug ? Create `an issue <https://github.com/nilearn/nilearn/issues>`_
  including a minimal runnable example to showcase it (preferably using Nilearn data).

* If you have ideas for new features, check it's in the scope of the project
  and feel free to open an issue to discuss it.

* You think documentation can be improved, you can directly send proposed
  improvements in `a pull request <https://github.com/nilearn/nilearn/pulls>`_.

* Want to contribute code ? Thank you very much! For new features, please be
  sure to create an issue first, to discuss whether it can be included and its
  specifications. If you have time to help dealing with known issues check
  `Good first issues <https://github.com/nilearn/nilearn/labels/Good%20first%20issue>`_
  to get started, `known bugs <https://github.com/nilearn/nilearn/labels/Bug>`_
  or `proposed enhancements <https://github.com/nilearn/nilearn/labels/Enhancement>`_.
  In any case, before sending code, be sure to ** read and follow the
  contribution guide below **. Otherwise we might ask quite a lot of revisions
  during the reviewing process.


How do we decide what codes goes in?
=====================================


Scope of the project
---------------------

Nilearn strives to develop open and powerful statistical analysis of
brain volumes (as produced by multiple modalities: MRI, PET, and others).
Its focus is to reach end users of the methods (as opposed to methods
developers).

Nilearn targets ease of use, but as Python code. In other words, we will
not add graphical user interfaces, but we want our code to be as easy to
understand as possible, with easy prototyping and debugging, even for
beginners in Python.

We are parsimonious in the way we add features to the project, as it
puts on weight. To assess new features, our main concern is their usefulness
to a number of our users. To make Nilearn high-quality and sustainable we also
weigh their benefits (i.e., new features, ease of use) with their cost (i.e.,
complexity of the code, runtime of the examples). As a rule of thumb:

* To be accepted, new features must be **in the scope of the project** and
  correspond to an established practice (typically as used in scientific
  publications)

* It must have a concrete use case, illustrated with a simple example in the
  nilearn documentation to teach it easily to end-users.

* It must be thouroughly tested, and respect coding conventions of the existing codebase.

* Features introducing new dependencies will generally not be accepted.

* Downloaders for new atlases are welcomed if they comes with an example.

* Downloaders for new datasets are usually discouraged. We will consider adding
  fetchers only for light datasets which are needed to demo and teach features.

Exhaustive criteria used in the review process are detailed in the contribution
guide below. Be sure to read and follow them so that your code can be accepted quickly.


Who makes decisions
--------------------

We strongly aim to be a community oriented project where decisions are
made based on consensus according to the criteria described above.
Decisions are made public, through discussion on issues and pull requests
in Github.

The decisions are made by the core-contributors, ie people with write
access to the repository, as listed :ref:`here <core_devs>`

How to contribute to nilearn
=============================

This project, hosted on https://github.com/nilearn/nilearn, is a community
effort, and everyone is welcome to contribute. We value very much your feedback
and opinion on features that should be improved or added. All discussions
are public and hold on relevant issues or pull requests. To discuss your matter,
please answer a relevant `issue <https://github.com/nilearn/nilearn/issues>`_
or open a new one.

The best way to contribute and to help the project is to start working on known
issues such as `Good first issues <https://github.com/nilearn/nilearn/labels/Good%20first%20issue>`_ ,
`known bugs <https://github.com/nilearn/nilearn/labels/Bug>`_ or
`proposed enhancements <https://github.com/nilearn/nilearn/labels/Enhancement>`_.

If an issue does not already exist for a potential contribution, we ask that
you first open an `issue <https://github.com/nilearn/nilearn/issues>`_ before
sending a :ref:`pull request` to discuss in advance scope and potential design
choices.


Contribution Guidelines
---------------------------

We ask every contributor to respect common guidelines. Those are inspired from
`scikit-learn
<https://scikit-learn.org/stable/developers/contributing.html#contributing-code>`_
and ensure Nilearn remains simple to understand, efficient and maintainable.
For example tests need to run quickly in order not to burden the development process.
To keep continuous integration efficient with our limited infrastructure, running
all the examples must lead to downloading a limited amount of data (gigabytes)
and execute in a reasonable amount of time (a few hours).

Those guidelines will hence be checked during reviewing process.


+--------------+-------------+---------------------------------------+
|              | Which PR ?  |        Guidelines                     |
+==============+=============+=======================================+
|              |             | - Clearly showcase benefits           |
|  Examples    | New features| - Run in a few minutes                |
|              |             | - Use light data from Nilearn         |
|              |             | - Renders well after build            |
+--------------+-------------+---------------------------------------+
|              |             | - Test type is adapted to behavior    |
|              |             | - Tests pass continuous integration   |
|              |  Bugfixes   | - Doesn't decrease coverage           |
|    Tests     | New features| - Fast, using small mocked data       |
|              |             | - Atomic (one per function) and seeded|
|              |             | - For Bugfixes: non-regression test   |
+--------------+-------------+---------------------------------------+
|              |             | - Variables, functions, arguments     |
|              |             | have clear and consistent names       |
|              |             | - Easy to read, PEP8                  |
| Coding Style |    Any      | - Clear docstring of public functions |
|              |             | - Low redundancy                      |
|              |             | - No new dependency                   |
|              |             | - Backward compatibility              |
+--------------+-------------+---------------------------------------+
|              |             | - Simple and didactic                 |
| Documentation|    Any      | - Links to relevant examples          |
|              |             | - Renders well after build            |
|              |             | - Doesn't include code                |
+--------------+-------------+---------------------------------------+
|    Other     |    Any      | - Add entry in "doc/whats_new.rst"     |
+--------------+-------------+---------------------------------------+

Contributing to the documentation
-------------------------------------------------

To build our documentation, we are using `sphinx <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_ for the main documentation and `sphinx-gallery <https://sphinx-gallery.github.io/stable/index.html>`_ for the example tutorials.
If you want to make changes to the example tutorials, please do the following :

1. First, ensure that you have installed sphinx and sphinx-gallery. You can install the requirements using ``nilearn/requirements-build-docs.txt``.
2. Fork the Nilearn repository and clone your fork.
3. Then go to ``nilearn/examples``
4. Make your changes using `reStructuredText files <https://www.sphinx-doc.org/en/2.0/usage/restructuredtext/basics.html>`_
5. You can now go to `nilearn/doc` and build the examples locally::

      make html-strict

   or, if you do not have make install (for instance under Windows)::

      python3 -m sphinx -b html -d _build/doctrees . _build/html

6. Visually review the output in ``nilearn/doc/_build/html/auto_examples/``. If all looks well and there were no errors, commit and push the changes.
7. You can now open a Pull Request from Nilearn's Pull Request page.

For more details about the Fork Clone Push worksflow, read here <https://guides.github.com/activities/forking/>_


TIPS : To reduce building time, we suggest you to use the ``filename_pattern`` to build just one specific file::

      python3 -m sphinx -D sphinx_gallery_conf.filename_pattern=plot_decoding_tutorial.py -b html -d _build/doctrees . _build/html


.. _git_repo:

Retrieving the latest code
---------------------------

We use `Git <http://git-scm.com/>`_ for version control and
`GitHub <https://github.com/>`_ for hosting our main repository. If you are
new on GitHub and don't know how to work with it, please first
have a look at `this <https://try.github.io/>`_ to get the basics.


You can check out the latest sources with the command::

    git clone git://github.com/nilearn/nilearn.git

or if you have write privileges::

    git clone git@github.com:nilearn/nilearn.git

Installing the latest code
---------------------------

In order to ensure that any code changes are reflected in your installation, navigate to your cloned Nilearn base directory and install using the following command::

    pip install -e .




Special case: How to contribute a dataset fetcher
--------------------------------------------------

The ``nilearn.datasets`` package provides functions to download some
neuroimaging datasets, such as ``fetch_haxby`` or
``fetch_atlas_harvard_oxford``. The goal is not to provide a comprehensive
collection of downloaders for the most widely used datasets, and this would be
outside the scope of this project. Rather, this package downloads data that is
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
such as ``get_dataset_dir`` to find a directory where the dataset is or will be
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


Special case: How to contribute an atlas
-----------------------------------------

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

In addition, the atlas will need to be called by a fetcher. For example, see `here <https://github.com/nilearn/nilearn/blob/master/nilearn/datasets/atlas.py>`__.

Finally, as with other features, please provide a test for your atlas.
Examples can be found `here
<https://github.com/nilearn/nilearn/blob/master/nilearn/datasets/tests/test_atlas.py>`__
