.. _maintenance_process:

============================
Nilearn maintenance process
============================

.. contents::
    :depth: 2
    :local:

Project Organization
====================

This section describes how the project is organized.

Issues
------

Nilearn uses `issues <https://github.com/nilearn/nilearn/issues>`_ for
tracking bugs, requesting potential features, and holding project discussions.

.. _issue_labels:

Labels
......

`Labels <https://github.com/nilearn/nilearn/labels>`_ are useful to
quickly sort `issues <https://github.com/nilearn/nilearn/issues>`_
and easily find what you are looking for in the issue tracker.

When `creating an issue
<https://github.com/nilearn/nilearn/issues/new/choose>`_, the user
is responsible for a very basic labeling categorizing the issue:

	- |Bug| for bug reports.
	- |Documentation| for documentation related questions or requests.
	- |Enhancement| for feature requests.

First of all, the user might have mislabeled the issue, in which case a member
of the :ref:`core_devs` team or :ref:`triage` needs to correct the labels.

In addition to these basic labels, we have many more labels which describes
in more detail a given issue. First, we try to describe the **estimated amount
of work** required to solve each issue:

	- |effort: high| The issue is likely to require a serious amount of work (more than a couple of days).
	- |effort: medium| The issue is likely to require a decent amount of work (in between a few hours and a couple days).
	- |effort: low| The issue is likely to require a small amount of work (less than a few hours).

We also try to quantify the **estimated impact** of the proposed change on the project:

	- |impact: high| Solving this issue will have a high impact on the project.
	- |impact: medium| Solving this issue will have a decent impact on the project.
	- |impact: low| Solving this issue will have a small impact on the project.

Finally, we also indicate the **priority level** of the issue:

	- |priority: high| The task is urgent and needs to be addressed as soon as possible.
	- |priority: medium| The task is important but not urgent and should be addressed over the next few months.
	- |priority: low| The task is not urgent and can be delayed.

Some issues---particular those which are low effort and low to medium priority---can serve as good starting project for
new contributors. We label these issues with the |Good first issue| label
which can be seen as an equivalent to a "very low effort" label. Because of
this, good first issues do not require a separate effort label.

Other labels can be used to describe further the topic of the issue:

	- |API| This issue is related to the Nilearn's API.
	- |code quality| This issue tackles code quality (code refactoring, PEP8...).
	- |Datasets| This issue is related to datasets or the :mod:`nilearn.datasets` module.
	- |Discussion| This issue is used to hold a general discussion on a specific topic where community feedback is desired (no need to specify effort, priority, or impact here).
	- |GLM| This issue is related to the :mod:`nilearn.glm` module.
	- |Infrastructure| This issue describes a problem with the project's infrastructure (CI/CD...).
	- |Installation| The issue describes a problem with the installation of Nilearn.
	- |Maintenance| This issue is related to maintenance work.
	- |Plotting| The issue is related to plotting functionalities.
	- |Testing| The issue is related to testing.
	- |Usage| This issue is a usage question and should have been posted on `neurostars <https://neurostars.org/>`_.

Finally, we use the following labels to indicate how the work on the issue
is going:

	- |in progress| Can be used to indicate that this issue is currently being investigated.
	- |next-release| Commonly used for tagging PRs, this can be used to indicate that this issue should be solved before the next release.
	- |stalled| This issue is currently stalled and has no recent activity. Use this label before closing due to inactivity.

.. |API| image:: https://img.shields.io/badge/-API-fef2c0.svg
.. |Bug| image:: https://img.shields.io/badge/-Bug-fc2929.svg
.. |code quality| image:: https://img.shields.io/badge/-code%20quality-09ef5a.svg
.. |Datasets| image:: https://img.shields.io/badge/-Datasets-fad8c7.svg
.. |Discussion| image:: https://img.shields.io/badge/-Discussion-bfe5bf.svg
.. |Documentation| image:: https://img.shields.io/badge/-Documentation-5319e7.svg
.. |effort: high| image:: https://img.shields.io/badge/-effort:%20high-e26051.svg
.. |effort: medium| image:: https://img.shields.io/badge/-effort:%20medium-ddad1a.svg
.. |effort: low| image:: https://img.shields.io/badge/-effort:%20low-77c940.svg
.. |Enhancement| image:: https://img.shields.io/badge/-Enhancement-fbca04.svg
.. |GLM| image:: https://img.shields.io/badge/-GLM-fce1c4.svg
.. |Good first issue| image:: https://img.shields.io/badge/-Good%20first%20issue-c7def8.svg
.. |impact: high| image:: https://img.shields.io/badge/-impact:%20high-1f1dc1.svg
.. |impact: medium| image:: https://img.shields.io/badge/-impact:%20medium-bac1fc.svg
.. |impact: low| image:: https://img.shields.io/badge/-impact:%20low-75eae6.svg
.. |in progress| image:: https://img.shields.io/badge/-in%20progress-ededed.svg
.. |Infrastructure| image:: https://img.shields.io/badge/-Infrastructure-0052cc.svg
.. |Installation| image:: https://img.shields.io/badge/-Installation-ba7030.svg
.. |Maintenance| image:: https://img.shields.io/badge/-Maintenance-fc918f.svg
.. |next-release| image:: https://img.shields.io/badge/-next--release-55c11f.svg
.. |Plotting| image:: https://img.shields.io/badge/-Plotting-5319e7.svg
.. |priority: high| image:: https://img.shields.io/badge/-priority:%20high-9e2409.svg
.. |priority: medium| image:: https://img.shields.io/badge/-priority:%20medium-FBCA04.svg
.. |priority: low| image:: https://img.shields.io/badge/-priority:%20low-c5def5.svg
.. |stalled| image:: https://img.shields.io/badge/-stalled-c2e0c6.svg
.. |Testing| image:: https://img.shields.io/badge/-Testing-50bac4.svg
.. |Usage| image:: https://img.shields.io/badge/-Usage-e99695.svg

.. _closing_policy:

Closing policy
..............

Usually we expect the issue's author to close the issue, but there are several
possible reasons for a community member to close an issue:

	- The issue has been solved: kindly asked the author whether the issue can be closed. In the absence of reply, close the issue after two weeks.
	- The issue is a usage question: label the issue with |Usage| and kindly redirect the author to `neurostars <https://neurostars.org/>`_. Close the issue afterwards.
	- The issue has no recent activity (no messages in the last three months): ping the author to see if the issue is still relevant. In the absence of reply, label the issue with |stalled| and close it after 2 weeks.

.. _pull request:

Pull Requests
---------------

We welcome pull requests from all community members, if they follow the
:ref:`contribution_guidelines` inspired from scikit learn conventions. (More
details on their process are available `here
<https://scikit-learn.org/stable/developers/contributing.html#contributing-code>`_)


How to make a release?
======================

This section describes how to make a new release of Nilearn. It is targeted to the specific case of Nilearn although it contains generic steps for packaging and distributing projects. More detailed information can be found on `packaging.python.org <https://packaging.python.org/guides/distributing-packages-using-setuptools/#id70>`_.

We assume that we are in a clean state where all the Pull Requests (PR) that we wish to include in the new release have been merged.
For example, make sure all deprecations that are supposed to be removed with this new version have been addressed. Furthermore, if this new release comes with dependency version bumps (Python, Numpy...), make sure to implement and test these changes beforehand. Ideally, these would have been done before such as to update the code base if necessary. Finally, make sure the documentation can be built correctly.

Prepare the release
-------------------

Switch to a new branch locally:

.. code-block:: bash

    git checkout -b REL-x.y.z


First we need to prepare the release by updating the file `nilearn/doc/whats_new.rst` to make sure all the new features, enhancements, and bug fixes are included in their respective sections.
We also need to write a "Highlights" section promoting the most important additions that come with this new release, and add the version tag just above the corresponding title:

.. code-block:: RST

    .. _vx.y.z:

    x.y.z
    =====
    **Released MONTH YEAR**

    HIGHLIGHTS
    ----------

    - Nilearn now includes functionality A
    - ...


Next, we need to bump the version number of Nilearn by updating the file `nilearn/version.py` with the new version number, that is edit the line:

.. code-block:: python

    __version__ = x.y.z.dev


to be:

.. code-block:: python

    __version__ = x.y.z


We also need to update the website news section by editing the file `nilearn/doc/themes/nilearn/layout.html`. The news section typically contains links to the last 3 releases that should look like:

.. code-block:: html

    <h4> News </h4>
        <ul>
            <li><p><strong>November 2020</strong>:
                <a href="whats_new.html#v0-7-0">Nilearn 0.7.0 released</a>
            </p></li>
            <li><p><strong>February 2020</strong>:
                <a href="whats_new.html#v0-6-2">Nilearn 0.6.2 released</a>
            </p></li>
            <li><p><strong>January 2020</strong>:
                <a href="whats_new.html#v0-6-1">Nilearn 0.6.1 released</a>
            </p></li>
        </ul>


Here, we should remove the last entry and add the new release on top of the list.

In addition, we can have a look at `MANIFEST.in` to check that all additional files that we want to be included or excluded from the release are indicated. Normally we shouldn't have to touch this file.

Add these changes and submit a PR:

.. code:: bash

    git add doc/whats_new.rst nilearn/version.py
    git commit -m "REL x.y.z"
    git push origin REL-x.y.z


Once the PR has been reviewed and merged, pull from master and tag the merge commit:

.. code:: bash

    git checkout master
    git pull upstream master
    git tag x.y.z
    git push upstream --tags


Build the distributions and upload them to Pypi
-----------------------------------------------

First of all we should make sure we don't include files that shouldn't be present:

.. code-block:: bash

    git checkout x.y.z


If the workspace contains a `dist` folder, make sure to clean it:

.. code-block:: bash

    rm -r dist


In order to build the binary wheel files, we need to install `wheel <https://pypi.org/project/wheel/>`_:

.. code-block:: bash

    pip install wheel


And, in order to upload to `Pypi`, we will use `twine <https://pypi.org/project/twine/>`_ that you can also install with `pip`:

.. code-block:: bash

    pip install twine


Build the source and binary distributions:

.. code-block:: bash

    python setup.py sdist bdist_wheel


This should add two files to the `dist` subfolder:

- one for the source distribution that should look like `PACKAGENAME-VERSION.tar.gz`
- one for the built distribution that should look like `PACKAGENAME-PACKAGEVERSION-PYTHONVERSION-PYTHONCVERSION-PLATFORM.whl`

Optionally, we can run some basic checks with `twine`:

.. code-block:: bash

    twine check dist/*


We are now ready to upload to `Pypi`. Note that you will need to have an `account on Pypi <https://pypi.org/account/register/>`_, and be added to the maintainers of `Nilearn <https://pypi.org/project/nilearn/>`_. If you satisfy these conditions, you should be able to run:

.. code-block:: bash

    twine upload dist/*


Once the upload is completed, make sure everything looks good on `Pypi <https://pypi.org/project/nilearn/>`_. Otherwise you will probably have to fix the issue and start over a new release with the patch number incremented.

At this point, we need to upload the binaries to GitHub and link them to the tag. To do so, go to the `Nilearn GitHub page <https://github.com/nilearn/nilearn/tags>`_ under the "Releases" tab, and edit the `x.y.z` tag by providing a description, and upload the distributions we just created (you can just drag and drop the files).


Build and deploy the documentation
----------------------------------

We now need to update the documentation:

.. code-block:: bash

    cd doc
    make install


This will build the documentation (beware, this is time consuming...) and push it to the `GitHub pages repo <https://github.com/nilearn/nilearn.github.io>`_.

Post-release
------------

At this point, the release has been made. We can now update the file `nilearn/version.py` and update the version number by increasing the patch number and appending `.dev`:

.. code-block:: python

    __version__ = x.y.(z+1).dev


We can also update the file `doc/whats_new.rst` by adding a title and the usual `New`, `Enhancements`, and `Bug Fixes` sections for the version currently under development:

.. code-block:: RST

    x.y.z+1.dev
    =========

    NEW
    ---

    Fixes
    -----

    Enhancements
    ------------

    .. _vx.y.z:

    x.y.z
    =====
    ...
