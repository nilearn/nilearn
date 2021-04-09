.. _contributing:

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
and ensure Nilearn remains simple to understand, efficient and maintanable.
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
|    Other     |    Any      | - Add entry in "doc/whats_new.py"     |
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
  related to its construction (modality, dataset, method), and if there are
  more than one maps, a description of each map.
- ``labels`` (list): a list of string labels corresponding to each atlas
  label, in the same (numerical) order as the atlas labels
- ``maps`` (list or string): the path to the nifti image, or a list of paths

In addition, the atlas will need to be called by a fetcher. For example, see `here <https://github.com/nilearn/nilearn/blob/master/nilearn/datasets/atlas.py>`__.

Finally, as with other features, please provide a test for your atlas.
Examples can be found `here
<https://github.com/nilearn/nilearn/blob/master/nilearn/datasets/tests/test_atlas.py>`__
