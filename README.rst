.. image:: https://img.shields.io/pypi/v/nilearn.svg
    :target: https://pypi.org/project/nilearn/
    :alt: Pypi Package

.. image:: https://img.shields.io/pypi/pyversions/nilearn.svg
    :target: https://pypi.org/project/nilearn/
    :alt: PyPI - Python Version

.. image:: https://github.com/nilearn/nilearn/workflows/build/badge.svg?branch=main&event=push
    :target: https://github.com/nilearn/nilearn/actions
    :alt: Github Actions Build Status

.. image:: https://codecov.io/gh/nilearn/nilearn/graph/badge.svg?token=KpYArSdyXv
    :target: https://codecov.io/gh/nilearn/nilearn
    :alt: Coverage Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8397156.svg
    :target: https://doi.org/10.5281/zenodo.8397156

.. image:: http://img.shields.io/twitter/follow/nilearn.svg
    :target: https://twitter.com/nilearn
    :alt: Twitter

.. image:: https://img.shields.io/mastodon/follow/109669703955432270?domain=https%3A%2F%2Ffosstodon.org%2F
    :target: https://fosstodon.org/@nilearn
    :alt: Mastodon

.. image:: https://img.shields.io/discord/711993354929569843
    :target: https://discord.gg/SsQABEJHkZ
    :alt: Discord




nilearn
=======

Nilearn enables approachable and versatile analyses of brain volumes. It provides statistical and machine-learning tools, with instructive documentation & friendly community.

It supports general linear model (GLM) based analysis and leverages the `scikit-learn <https://scikit-learn.org>`_ Python toolbox for multivariate statistics with applications such as predictive modelling, classification, decoding, or connectivity analysis.

Important links
===============

- Official source code repo: https://github.com/nilearn/nilearn/
- HTML documentation (stable release): https://nilearn.github.io/

Install
=======

Latest release
--------------

**1. Setup a virtual environment**

We recommend that you install ``nilearn`` in a virtual Python environment,
either managed with the standard library ``venv`` or with ``conda``
(see `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ for instance).
Either way, create and activate a new python environment.

With ``venv``:

.. code-block:: bash

    python3 -m venv /<path_to_new_env>
    source /<path_to_new_env>/bin/activate

Windows users should change the last line to ``\<path_to_new_env>\Scripts\activate.bat`` in order to activate their virtual environment.

With ``conda``:

.. code-block:: bash

    conda create -n nilearn python=3.9
    conda activate nilearn

**2. Install nilearn with pip**

Execute the following command in the command prompt / terminal
in the proper python environment:

.. code-block:: bash

    python -m pip install -U nilearn

Development version
-------------------

Please find all development setup instructions in the
`contribution guide <https://nilearn.github.io/stable/development.html#setting-up-your-environment>`_.

Check installation
------------------

Try importing nilearn in a python / iPython session:

.. code-block:: python

    import nilearn

If no error is raised, you have installed nilearn correctly.

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

The required dependencies to use the software are listed in the file `pyproject.toml <https://github.com/nilearn/nilearn/blob/main/pyproject.toml>`_.

If you are using nilearn plotting functionalities or running the examples, matplotlib >= 3.3.0 is required.

Some plotting functions in Nilearn support both matplotlib and plotly as plotting engines.
In order to use the plotly engine in these functions, you will need to install both plotly and kaleido, which can both be installed with pip and anaconda.

If you want to run the tests, you need pytest >= 6.0.0 and pytest-cov for coverage reporting.

Development
===========

Detailed instructions on how to contribute are available at
https://nilearn.github.io/stable/development.html
