Installing Nilearn
==================

There are different ways to install Nilearn:

* :ref:`Install the latest official release <install_official_release>`. This
  is the best approach for most users.

* :ref:`Building the package from source <setup_development_environment>`.
  This is mainly needed by users who wish to contribute to the project, as this allows
  to install an editable version of the project.

.. tab-set::

    .. tab-item:: pip

        We assume that you are using ``venv`` to create a virtual environment.

        .. code-block:: bash

            python3 -m venv
            source source .venv/bin/activate
            pip install nilearn

    .. tab-item:: uv

        .. code-block:: bash

            uv venv -p 3.14 #

    .. tab-item:: conda


        .. code-block:: bash

            conda create -n nilearn python=3.14
            conda activate nilearn
