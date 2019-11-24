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
