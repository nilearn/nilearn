.. _maintenance_process:

===========
Maintenance
===========

Project Organization
====================

This section describes how the project is organized.

Issues
------

Nilearn uses :nilearn-gh:`issues <issues>` for
tracking bugs, requesting potential features, and holding project discussions.

.. _issue_labels:

Labels
......

:nilearn-gh:`Labels <labels>` are useful to
quickly sort :nilearn-gh:`issues <issues>`
and easily find what you are looking for in the issue tracker.

When :nilearn-gh:`creating an issue <issues/new/choose>`, the user
is responsible for a very basic labeling categorizing the issue:

- |Bug| for bug reports.
- |Documentation| for documentation related questions or requests.
- |Enhancement| for feature requests.

First of all, the user might have mislabeled the issue, in which case a member
of the :ref:`core_devs` team needs to correct the labels.

In addition to these basic labels, we have many more labels which describes
in more detail a given issue. First, we try to describe the **estimated amount
of work** required to solve each issue:

- |Effort: high| The issue is likely to require a serious amount of work (more than a couple of days).
- |Effort: medium| The issue is likely to require a decent amount of work (in between a few hours and a couple days).
- |Effort: low| The issue is likely to require a small amount of work (less than a few hours).

We also try to quantify the **estimated impact** of the proposed change on the project:

- |Impact: high| Solving this issue will have a high impact on the project.
- |Impact: medium| Solving this issue will have a decent impact on the project.
- |Impact: low| Solving this issue will have a small impact on the project.

Finally, we also indicate the **priority level** of the issue:

- |Priority: high| The task is urgent and needs to be addressed as soon as possible.
- |Priority: medium| The task is important but not urgent and should be addressed over the next few months.
- |Priority: low| The task is not urgent and can be delayed.

Some issues---particular those which are low effort and low to medium priority---can serve as good starting project for
new contributors. We label these issues with the |Good first issue| label
which can be seen as an equivalent to a "very low effort" label. Because of
this, good first issues do not require a separate effort label.

Some other labels can be used to describe further the topic of the issue:

-   |API| This issue is related to the Nilearn's API.
-   |Code quality| This issue tackles code quality (code refactoring, PEP8...).
-   |Datasets| This issue is related to datasets or the :mod:`nilearn.datasets` module.
-   |Discussion| This issue is used to hold a general discussion on a specific topic
    where community feedback is desired (no need to specify effort, priority, or impact here).
-   |GLM| This issue is related to the :mod:`nilearn.glm` module.
-   |Infrastructure| This issue describes a problem with the project's infrastructure (CI/CD...).
-   |Installation| The issue describes a problem with the installation of Nilearn.
-   |Maintenance| This issue is related to maintenance work.
-   |Plotting| The issue is related to plotting functionalities.
-   |Testing| The issue is related to testing.
-   |Usage| This issue is a usage question and should have been posted on :neurostars:`neurostars <>`.

For a complete list of all issue labels that can be used to describe and their description,
see `this page <https://github.com/nilearn/nilearn/labels>`_

.. |API| image:: https://img.shields.io/badge/-API-fef2c0.svg
.. |Bug| image:: https://img.shields.io/badge/-Bug-fc2929.svg
.. |Code quality| image:: https://img.shields.io/badge/-code%20quality-09ef5a.svg
.. |Datasets| image:: https://img.shields.io/badge/-Datasets-fad8c7.svg
.. |Discussion| image:: https://img.shields.io/badge/-Discussion-bfe5bf.svg
.. |Documentation| image:: https://img.shields.io/badge/-Documentation-5319e7.svg
.. |Effort: high| image:: https://img.shields.io/badge/-effort:%20high-e26051.svg
.. |Effort: medium| image:: https://img.shields.io/badge/-effort:%20medium-ddad1a.svg
.. |Effort: low| image:: https://img.shields.io/badge/-effort:%20low-77c940.svg
.. |Enhancement| image:: https://img.shields.io/badge/-Enhancement-fbca04.svg
.. |GLM| image:: https://img.shields.io/badge/-GLM-fce1c4.svg
.. |Good first issue| image:: https://img.shields.io/badge/-Good%20first%20issue-c7def8.svg
.. |Impact: high| image:: https://img.shields.io/badge/-impact:%20high-1f1dc1.svg
.. |Impact: medium| image:: https://img.shields.io/badge/-impact:%20medium-bac1fc.svg
.. |Impact: low| image:: https://img.shields.io/badge/-impact:%20low-75eae6.svg
.. |Infrastructure| image:: https://img.shields.io/badge/-Infrastructure-0052cc.svg
.. |Installation| image:: https://img.shields.io/badge/-Installation-ba7030.svg
.. |Maintenance| image:: https://img.shields.io/badge/-Maintenance-fc918f.svg
.. |Plotting| image:: https://img.shields.io/badge/-Plotting-5319e7.svg
.. |Priority: high| image:: https://img.shields.io/badge/-priority:%20high-9e2409.svg
.. |Priority: medium| image:: https://img.shields.io/badge/-priority:%20medium-FBCA04.svg
.. |Priority: low| image:: https://img.shields.io/badge/-priority:%20low-c5def5.svg
.. |Testing| image:: https://img.shields.io/badge/-Testing-50bac4.svg
.. |Usage| image:: https://img.shields.io/badge/-Usage-e99695.svg

.. _closing_policy:

Closing policy
..............

Usually we expect the issue's author to close the issue, but there are several
possible reasons for a community member to close an issue:

-   The issue has been solved: kindly asked the author whether the issue can be closed.
    In the absence of reply, close the issue after two weeks.
-   The issue is a usage question: label the issue with |Usage|
    and kindly redirect the author to :neurostars:`neurostars <>`.
    Close the issue afterwards.
-   The issue has no recent activity (no messages in the last three months):
    ping the author to see if the issue is still relevant.
    In the absence of reply, label the issue with ``stalled`` and close it after 2 weeks.

.. _pull request:

Pull Requests
---------------

We welcome pull requests from all community members, if they follow the
:ref:`contribution_guidelines` inspired from scikit learn conventions. (More
details on their process are available
:sklearn:`here <developers/contributing.html#contributing-code>`).

Using tox
=========

`Tox <https://tox.wiki/en/4.23.2/>`_ is set
to facilitate testing and managing environments during development
and ensure that the same commands can easily be run locally and in CI.

Install it with:

.. code-block:: bash

    pip install tox

You can set up certain environment or run certain command by calling ``tox``.

Calling ``tox`` with no extra argument will simply run
all the default commands defined in the tox configuration (``tox.ini``).

Use ``tox list`` to view all environment descriptions.

Use ``tox run`` to run a specific environment.

Example

.. code-block:: bash

    tox run -e lint

Some environments allow passing extra argument:

.. code-block:: bash

    # only run ruff
    tox run -e lint -- ruff

    # only run some tests
    tox -e test_plotting -- nilearn/glm/tests/test_contrasts.py

You can also run any arbitrary command in a given environment with ``tox exec``:

.. code-block:: bash

    tox exec -e test_latest -- python -m pytest nilearn/_utils/tests/test_data_gen.py


How to make a release?
======================

This section describes how to make a new release of Nilearn.
It is targeted to the specific case of Nilearn although it contains generic steps
for packaging and distributing projects.
More detailed information can be found on
`packaging.python.org <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_.

The packaging specification is contained in
`pyproject.toml <https://github.com/nilearn/nilearn/blob/main/pyproject.toml>`_.
We use ``hatchling`` and ``hatch-vcs``
as described in these `guidelines <https://effigies.gitlab.io/posts/python-packaging-2023/>`_
to build the sdist, wheel, and extract version number from the git tag.

We assume that we are in a clean state where all the Pull Requests (PR)
that we wish to include in the new release have been merged.

Prepare code for the release
----------------------------

The repository should be checked and updated in preparation for the release.

One thing that **must** be done before the release is made is
to update ``deprecated``, ``versionchanged`` and ``versionadded`` directives
from the current ``[x.y.z].dev`` tag to the new version number.
These directives are added in a function's docstring to indicate the version number,
when, say, a new parameter is added or deprecated.

For example, if a parameter ``param2`` was added in version ``x.y.z``, the docstring should be updated to:

.. code-block:: python

    def my_function(param1, param2):
        """
        Parameters
        ----------
        param1 : type
            Description of param1.

        param2 : type
            Description of param2.

        .. versionadded:: x.y.z

        Returns
        -------
        output : type
            Description of the output.
        """
        ...

Additionally, make sure all deprecations that are supposed to be removed with this new version have been addressed.

If this new release comes with dependency version bumps (Python, Numpy...),
make sure to implement and test these changes beforehand.
Ideally, these would have been done before such as to update the code base if necessary.
Finally, make sure the documentation can be built correctly.

Prepare the release
-------------------

Switch to a new branch locally:

.. code-block:: bash

    git checkout -b REL-x.y.z


First we need to prepare the release by updating the file ``nilearn/doc/changes/latest.rst``
to make sure all the new features, enhancements, and bug fixes are included in their respective sections.

Then we need to make sure that all the entries in each section of the changelog
in ``nilearn/doc/changes/latest.rst``:

-   have a label,
-   are sorted by their "label" alphabetically
-   and are followed by an empty line.

For example::

    - :bdg-success:`API` ...

    - :bdg-dark:`Code` ...

    - :bdg-info:`Plotting` ...

We also need to write a "Highlights" section promoting the most important additions that come with this new release.
Finally, we need to change the title from ``x.y.z.dev`` to ``x.y.z``:

.. code-block:: RST

   .. currentmodule:: nilearn

   .. include:: names.rst

   x.y.z
   =====

   **Released MONTH YEAR**

   HIGHLIGHTS
   ----------

   - Nilearn now includes functionality A
   - ...

We must also ensure that every entry in ``nilearn/doc/changes/latest.rst``
starts with a "badge" (see the :ref:`changelog` section).

Once we have made all the necessary changes to ``nilearn/doc/changes/latest.rst``,
we should rename it into ``nilearn/doc/changes/x.y.z.rst``,
where ``x.y.z`` is the corresponding version number.

We then need to update ``nilearn/doc/changes/whats_new.rst`` and replace:

.. code-block:: RST

   .. _latest:
   .. include:: latest.rst

By:

.. code-block:: RST

   .. _vx.y.z:
   .. include:: x.y.z.rst


Add these changes and submit a PR:

.. code:: bash

    git add doc/changes/
    git commit -m "REL x.y.z"
    git push origin REL-x.y.z


Once the PR has been reviewed and merged, pull from master and tag the merge commit:

.. code:: bash

    git checkout main
    git pull upstream main
    git tag x.y.z
    git push upstream --tags

.. note::

    When building the distribution as described below, ``hatch-vcs``, defined in ``pyproject.toml``,
    extracts the version number using this tag and writes it to a ``_version.py`` file.


Build the distributions and upload them to Pypi
-----------------------------------------------

First of all we should make sure we don't include files that shouldn't be present:

.. code-block:: bash

    git checkout x.y.z


If the workspace contains a ``dist`` folder, make sure to clean it:

.. code-block:: bash

    rm -r dist


In order to build the binary wheel files, we need to install `build <https://pypi.org/project/build/>`_:

.. code-block:: bash

    pip install build


And, in order to upload to ``Pypi``, we will use `twine <https://pypi.org/project/twine/>`_ that you can also install with ``pip``:

.. code-block:: bash

    pip install twine


Build the source and binary distributions:

.. code-block:: bash

    python -m build


This should add two files to the ``dist`` subfolder:

-   one for the source distribution that should look like ``PACKAGENAME-VERSION.tar.gz``
-   one for the built distribution
    that should look like ``PACKAGENAME-PACKAGEVERSION-PYTHONVERSION-PYTHONCVERSION-PLATFORM.whl``

This will also update ``_version.py``.

Optionally, we can run some basic checks with ``twine``:

.. code-block:: bash

    twine check dist/*


We are now ready to upload to ``Pypi``. Note that you will need to have an `account on Pypi <https://pypi.org/account/register/>`_, and be added to the maintainers of `Nilearn <https://pypi.org/project/nilearn/>`_. If you satisfy these conditions, you should be able to run:

.. code-block:: bash

    twine upload dist/*

Once the upload is completed, make sure everything looks good on `Pypi <https://pypi.org/project/nilearn/>`_.
Otherwise you will probably have to fix the issue and start over a new release with the patch number incremented.


Github release
--------------

At this point, we need to upload the binaries to GitHub and link them to the tag.
To do so, go to the :nilearn-gh:`Nilearn GitHub page <tags>` under the "Releases" tab,
and edit the ``x.y.z`` tag by providing a description,
and upload the distributions we just created (you can just drag and drop the files).


Build of stable docs
--------------------

Once the new tagged github release is made following the step above,
the Github Actions workflow ``release-docs.yml`` will be triggered automatically
to build the stable docs and push them
to our github pages repository ``nilearn/nilearn.github.io``.
The workflow can also be triggered manually from the Actions tab.


Build and deploy the documentation manually
-------------------------------------------

.. note::

    This step is now automated as described above.
    If there is a need to run it manually please follow the instructions below.


Before building the documentation, make sure that the following LaTeX
dependencies are installed on your system:

- `dvipng <https://ctan.org/pkg/dvipng>`_
- `texlive-latex-base <https://ctan.org/pkg/latex-base>`_
- `texlive-latex-extra <https://packages.debian.org/sid/texlive-latex-extra>`_

You can check if each package is installed by using
``command -v <command-name>`` as in:

.. code-block:: bash

    command -v dvipng

If the package is installed, then the path to its location on your system will
be returned. Otherwise, you can install using your system's package manager or
from source, for example:

.. code-block:: bash

    wget https://mirrors.ctan.org/dviware/dvipng.zip
    unzip dvipng.zip
    cd dvipng
    ./configure
    make
    make install

See available linux distributions of texlive-latex-base and texlive-latex-extra:

- https://pkgs.org/search/?q=texlive-latex-base
- https://pkgs.org/search/?q=texlive-latex-extra

We now need to update the documentation.
We let tox handle creating virtual env and install dependencies.

.. warning::

    The doc build is done with the minimum python version supported by Nilearn.

.. code-block:: bash

    export VERSIONTAG=$(git describe --tags --abbrev=0)
    pip install tox
    tox run --colored yes --list-dependencies -e doc -- install


This will build the documentation (beware, this is time consuming...)
and push it to the `GitHub pages repo <https://github.com/nilearn/nilearn.github.io>`_.

Post-release
------------

At this point, the release has been made.

We also need to create a new file ``doc/changes/latest.rst`` with a title
and the usual ``New``, ``Enhancements``, ``Bug Fixes``, and ``Changes``
sections for the version currently under development:

.. code-block:: RST

   .. currentmodule:: nilearn

   .. include:: names.rst

   x.y.z+1.dev
   =========

   NEW
   ---

   Fixes
   -----

   Enhancements
   ------------

   Changes
   -------

Finally, we need to include this new file in ``doc/changes/whats_new.rst``:

.. code-block:: RST

   .. _latest:
   .. include:: latest.rst
