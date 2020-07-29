============================
Nilearn development process
============================

.. contents::
    :depth: 2
    :local:

How to help?
=============

* You are new to python and you don't know how to do xy

   - Create a question on `neurostars <https://neurostars.org/>`_

* If you discovered a bug, but don't know how to fix it

   - Create `an issue <https://github.com/nilearn/nilearn/issues>`_

* If you discovered a bug and know how to fix it, but don't know how to
  get your code onto github (ie, you have some Python experience but have
  never have used git/github, or have never written tests)

    - Learn git and github: http://try.github.io/
    - Learn what tests are how to run them locally
      (https://docs.pytest.org)
    - Learn how to write doc/examples and build them locally
      https://sphinx-gallery.github.io/

* You want to contribute code

    - See below


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
puts on weight. Criteria include:

* It must be in the scope of the project
* It must correspond to an established practice (typically as used in
  scientific publications)
* It must have a concrete use case that can be demo-ed simply with nilearn:
  an example, on real data, understandable by end-users.

Part of the decision will also be about weighing the benefits (i.e., new
features or ease of use for the users) with the cost (i.e., complexity of
the code, runtime of the examples).

In practice:

* The new feature must be demoed in an example in a way that shows its
  benefit to new users.
* Because our infrastructure is limited, running all the examples must
  lead to downloading a limited amount of data (gigabytes) and execute
  in a reasonable amount of time (a few hours)
* The new feature must be thoroughly tested (it should not decrease
  code coverage)
* The new feature may not introduce a new dependency

Special cases:

* A downloader for a new atlas: we are currently being very lenient for this:
  if the atlas is published and can be used in an example, we will accept
  the pull request (but see below for specifics).
* A downloader for a new dataset: the larger the dataset is, the less
  likely we are to consider including it. Datasets are meant to demo and
  teach features, rather than be the basis of research.

How to contribute a feature
----------------------------

To contribute a feature, first create an issue, in order to discuss
whether the feature can be included or not, and the specifications of
this feature. Once agreed on the feature, send us a pull request.

There are specific guidelines about how to write code for the project.
They can be found in the contributors guide, below.

Special case: How to contribute a dataset fetcher
.................................................

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
.............................................

We want atlases in nilearn to be internally consistent. Specifically,
your atlas object should have three attributes (as with the existing
atlases):

- ``description`` (bytes): A text description of the atlas. This should be
  brief but thorough, describing the source (paper), relevant information
  related to its construction (modality, dataset, method), and if there are
  more than one maps, a description of each map.
- ``labels`` (list): a list of string labels corresponding to each atlas
  label, in the same (numerical) order as the atlas labels
- ``maps`` (list or string): the path to the nifti image, or a list of paths

In addition, the atlas will need to be called by a fetcher. For example, see `here <https://github.com/nilearn/nilearn/blob/master/nilearn/datasets/atlas.py>`__.

Finally, as with other features, please provide a test for your atlas.
Examples can be found `here
<https://github.com/nilearn/nilearn/blob/master/nilearn/datasets/tests/test_atlas.py>`__

Who makes decisions
--------------------

We strongly aim to be a community oriented project where decisions are
made based on consensus according to the criteria described above.
Decisions are made public, through discussion on issues and pull requests
in Github.

The decisions are made by the core-contributors, ie people with write
access to the repository, as listed :ref:`here <core_devs>`

If there are open questions, final decisions are made by the Temporary
Benevolent Dictator, currently Gaël Varoquaux.

.. include:: ../CONTRIBUTING.rst
