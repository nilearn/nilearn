.. image:: https://img.shields.io/pypi/v/nilearn.svg
    :target: https://pypi.org/project/nilearn/
    :alt: Pypi Package

.. image:: https://img.shields.io/pypi/pyversions/nilearn.svg
    :target: https://pypi.org/project/nilearn/
    :alt: PyPI - Python Version

.. image:: https://github.com/nilearn/nilearn/actions/workflows/build-docs.yml/badge.svg
    :target: https://github.com/nilearn/nilearn/actions/workflows/build-docs.yml
    :alt: Github Actions Doc Build Status

.. image:: https://github.com/nilearn/nilearn/actions/workflows/test_with_tox.yml/badge.svg?branch=main&event=push
    :target: https://github.com/nilearn/nilearn/actions/workflows/test_with_tox.yml
    :alt: Github Actions Test Status

.. image:: https://codecov.io/gh/nilearn/nilearn/graph/badge.svg?token=KpYArSdyXv
    :target: https://app.codecov.io/gh/nilearn/nilearn
    :alt: Coverage Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8397156.svg
    :target: https://doi.org/10.5281/zenodo.8397156

.. image:: https://img.shields.io/badge/bluesky-Follow_nilearn-blue?logo=bluesky
    :target: https://bsky.app/profile/nilearn.bsky.social
    :alt: Bluesky

.. image:: https://img.shields.io/youtube/channel/subscribers/UCU6BMAi2zOhNFnDkbdevmPw
    :target: https://www.youtube.com/@nilearnevents5116
    :alt: YouTube Channel Subscribers

.. image:: https://img.shields.io/mastodon/follow/109669703955432270?domain=https%3A%2F%2Ffosstodon.org%2F
    :target: https://fosstodon.org/@nilearn
    :alt: Mastodon

.. image:: https://img.shields.io/discord/711993354929569843
    :target: https://discord.com/invite/SsQABEJHkZ
    :alt: Discord

nilearn
=======

Nilearn enables approachable and versatile analyses of brain volumes and surfaces.
It provides statistical and machine-learning tools, with instructive documentation & friendly community.

It supports general linear model (GLM) based analysis
and leverages the `scikit-learn <https://scikit-learn.org>`_ Python toolbox
for multivariate statistics with applications
such as predictive modeling, classification, decoding, or connectivity analysis.

Important links
===============

- Official source code repo: https://github.com/nilearn/nilearn/
- HTML documentation (stable release): https://nilearn.github.io/

Install
=======

Latest release
--------------

The easiest way to install ``nilearn`` is using pip.
Execute the following command in the command prompt / terminal
in the proper python environment:

.. prompt:: bash

    python -m pip install nilearn

Please find all installation instructions in the
`on our install page <https://nilearn.github.io/dev/install.html>`_.

Development version
-------------------

Please find all development setup instructions in the
`contribution guide <https://nilearn.github.io/stable/development.html#setting-up-your-environment>`_.

Drop-in Hours
=============

The Nilearn team organizes regular online drop-in hours to answer questions,
discuss feature requests, or have any Nilearn-related discussions. Nilearn
drop-in hours occur *every Wednesday from 4pm to 5pm UTC*, and we make sure that at
least one member of the core-developer team is available. These events are held
on `Jitsi Meet <https://meet.jit.si/nilearn-drop-in-hours>`_ and are fully open,
anyone is welcome to join!
For more information and ways to engage with the Nilearn team see
`How to get help <https://nilearn.github.io/stable/development.html#how-to-get-help>`_.

Dependencies
============

The required dependencies to use the software are listed
in the file `pyproject.toml <https://github.com/nilearn/nilearn/blob/main/pyproject.toml>`_.

If you are using nilearn plotting functionalities or running the examples, matplotlib is required.

Some plotting functions in Nilearn support both matplotlib and plotly as plotting engines.
In order to use the plotly engine in these functions,
you will need to install both plotly and kaleido, which can both be installed with pip and anaconda.

If you want to run the tests, you need pytest and pytest-cov for coverage reporting.

Development
===========

Detailed instructions on how to contribute are available at
https://nilearn.github.io/stable/development.html
