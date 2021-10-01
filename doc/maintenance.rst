.. _maintenance_process:

============================
Nilearn maintenance process
============================

.. contents::
    :depth: 2
    :local:

Project Organization
======================

Issues
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
