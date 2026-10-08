Installing Nilearn
==================

There are different ways to install Nilearn:

* Install the latest official release (see below).
  This is the best approach for most users.

* :ref:`Building the package from source <setup_development_environment>`.
  This is mainly needed by users who wish to contribute to the project,
  as this allows to install an editable version of the project.

.. _virtual_env:

Setup a virtual environment
---------------------------

We recommend that you install ``nilearn`` in a virtual Python environment,
either managed with the standard library ``venv``, `uv <https://docs.astral.sh/uv/>`_
or with ``conda`` (see `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ for instance).
Either way, create and activate a new Python environment.

.. tab-set::

    .. tab-item:: Linux / MacOS
        :class-label: tab-6

        .. tab-set::

            .. tab-item:: pip
                :class-label: tab-4

                .. prompt:: bash

                    python3 -m venv 'venv'
                    source venv/bin/activate

            .. tab-item:: uv
                :class-label: tab-4

                Note that ``uv`` allows you to choose the python version.

                .. prompt:: bash

                    uv venv 'venv' --python 3.14
                    source venv/bin/activate

            .. tab-item:: conda
                :class-label: tab-4

                Note that ``conda`` allows you to choose the python version.

                .. prompt:: bash

                    conda create --name nilearn python=3.14 pip
                    conda activate nilearn

    .. tab-item:: Windows
        :class-label: tab-6

        .. tab-set::

            .. tab-item:: pip
                :class-label: tab-4

                .. prompt:: powershell

                    python3 -m venv 'venv'
                    venv\Scripts\activate

            .. tab-item:: uv
                :class-label: tab-4

                Note that ``uv`` allows you to choose the Python version.

                .. prompt:: powershell

                    uv venv 'venv' --python 3.14
                    venv\Scripts\activate

            .. tab-item:: conda
                :class-label: tab-4

                Note that ``conda`` allows you to choose the Python version.

                .. prompt:: powershell

                    conda create --name nilearn python=3.14 pip
                    conda activate nilearn

Installing Nilearn
------------------

You can then install Nilearn.
Nilearn comes in different flavor.
On top of **base** Nilearn installation,
you can add ``matplotlib`` as optional dependency for **static** visualizations,
and ``matplotlib`` as well as ``plotly`` as optional dependencies
for both static and **interactive** visualizations.

.. admonition:: Important

    To be able to save images with plotly,
    make sure that Google Chrome is installed!
    You can install a compatible Chrome version using
    the ``kaleido_get_chrome`` command in command line or
    ``kaleido.get_chrome_sync()`` function
    in Python:

    .. code-block:: python

        import kaleido

        kaleido.get_chrome_sync()

.. tab-set::

    .. tab-item:: base
        :class-label: tab-4

        .. tab-set::

            .. tab-item:: pip / conda
                :class-label: tab-4

                .. code-block:: bash

                    pip install nilearn

            .. tab-item:: uv
                :class-label: tab-4

                .. code-block:: bash

                    uv pip install nilearn

    .. tab-item:: static
        :class-label: tab-4

        .. tab-set::

            .. tab-item:: pip / conda
                :class-label: tab-4

                .. code-block:: bash

                    pip install 'nilearn[plotting]'

            .. tab-item:: uv
                :class-label: tab-4

                .. code-block:: bash

                    uv pip install 'nilearn[plotting]'

    .. tab-item:: interactive
        :class-label: tab-4

        .. tab-set::

            .. tab-item:: pip / conda
                :class-label: tab-4

                .. code-block:: bash

                    pip install 'nilearn[plotting,plotly]'
                    python -c "import kaleido;  kaleido.get_chrome_sync()"

            .. tab-item:: uv
                :class-label: tab-4

                .. code-block:: bash

                    uv pip install 'nilearn[plotting,plotly]'
                    uv run 'python -c "import kaleido;  kaleido.get_chrome_sync()"'

Note that Nilearn also optionally supports `rich <https://rich.readthedocs.io/en/latest/introduction.html>`_
to get prettier log output and download progress bar.
Simply install ``rich`` to benefit from those.

.. code-block::

    pip install rich

Check installation
------------------

From a terminal window:

.. code-block:: bash

    python3 -c 'import nilearn; print(nilearn.__version__)'

Or try importing Nilearn in a Python / IPython session.

.. code-block:: python

    import nilearn

If no error is raised, you have installed Nilearn correctly.
