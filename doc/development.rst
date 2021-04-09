============================
Nilearn development process
============================

.. contents::
    :depth: 2
    :local:

# Ideally this should go directly on the web pag

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

.. include:: ../CONTRIBUTING.rst
