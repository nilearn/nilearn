.. _contributing:

How to contribute to nilearn
=============================

This project is a community effort, and everyone is welcome to
contribute.

The project is hosted on https://github.com/nilearn/nilearn

The best way to contribute and to help the project is to start working on known
issues.
See `Good first issues <https://github.com/nilearn/nilearn/labels/Good%20first%20issue>`_ to get
started.

If an issue does not already exist for a potential contribution, we ask that
you first open an `issue <https://github.com/nilearn/nilearn/issues>`_ before
sending a :ref:`pull request`.

Opening an issue
------------------

Nilearn uses issues for tracking bugs, requesting potential features, and
holding project discussions.

Core developers can assign labels on issues, such as:

- |Discussion| These issues discuss ongoing discussions on the project where community feedback is requested.
- |Enhancement| These issues discuss potential enhancements or additions to the project.
- |Bug| These issues detail known bugs in the Nilearn code base.

.. |Discussion| image:: https://img.shields.io/badge/-Discussion-bfe5bf.svg
.. |Enhancement| image:: https://img.shields.io/badge/-Enhancement-fbca04.svg
.. |Bug| image:: https://img.shields.io/badge/-Bug-fc2929.svg

.. _pull request:

Pull Requests
---------------

We welcome pull requests from all community members.
We follow the same conventions as scikit-learn. You can find the recommended process to submit code in the
`scikit-learn guide to contributing code
<https://scikit-learn.org/stable/developers/contributing.html#contributing-code>`_.

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

Coding guidelines
------------------

Nilearn follows the coding conventions used by scikit-learn. `Please read them
<http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines>`_
before you start implementing your changes.

Documentation
--------------

To build our documentation, we are using `sphinx <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_ for the main documentation and `sphinx-gallery <https://sphinx-gallery.github.io/stable/index.html>`_ for the example tutorials.
If you want to make changes to the example tutorials, please do the following :

1. First, ensure that you have installed sphynx and sphynx gallery
2. Clone the nilearn repostiory and go to ``nilearn/examples``
3. Make your changes using `reStructuredText files <http://docutils.sourceforge.net/rst.html>`_
4. You can now go to `nilearn/doc` and build the examples locally:

``python3 -m sphinx -b html -d _build/doctrees . _build/html``

5. Once you are correct with the html outputs in ``nilearn/doc/_build/html/auto_examples/``, you are good to submit the PR!

TIPS : To reduce building time, we suggest you to use the ``filename_pattern`` to build just one specific file.

``python3 -m sphinx -D sphinx_gallery_conf.filename_pattern=plot_decoding_tutorial.py -b html -d _build/doctrees . _build/html``
