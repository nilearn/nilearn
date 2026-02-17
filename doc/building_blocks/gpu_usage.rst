.. _gpu_usage:

=============================================================
Leverage GPUs for your neuroimaging machine-learning pipeline
=============================================================

Nilearn is dependent on scikit-learn for its machine learning tools.
With the recent developments in scikit-learn, there is now
:sklearn:`a growing list of estimators <modules/array_api.html#support-for-array-api-compatible-inputs>`
that support GPU acceleration via the Array API.

This means that you can now also use GPUs to speed up your neuroimaging
machine-learning pipelines in nilearn.

We will demonstrate this speedup using the :ref:`sphx_glr_auto_examples_02_decoding_plot_miyawaki_encoding.py` example,
which implements an encoding model using
:sklearn:`Ridge regression <linear_model/ridge.html>` to predict fMRI data
from visual stimuli.

You would need to have a compatible GPU and the latest pre-release version of scikit-learn to run the code here:

.. code-block:: bash

    pip install -q -U --pre --extra-index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn

In addition, you would need to have either `install CuPy
<https://docs.cupy.dev/en/stable/install.html>`_ or
`PyTorch <https://pytorch.org/get-started/locally/>`_
to run the GPU-accelerated code.

Furthermore, as mentioned in :sklearn:`scikit-learn docs <modules/array_api.html#array-api-support-experimental>`
you would also need to do the following:

.. code-block:: python

    import os
    from sklearn import set_config

    os.environ["SCIPY_ARRAY_API"] = "1"
    set_config(array_api_dispatch=True)


Now we can proceed to load and prepare the data as in the original example:

.. literalinclude:: ../../examples/02_decoding/plot_miyawaki_encoding.py
    :start-after: # data, clean and mask it.
    :end-before: # %%

.. literalinclude:: ../../examples/02_decoding/plot_miyawaki_encoding.py
    :start-after: # Now we can load the data set:
    :end-before: # %%

.. literalinclude:: ../../examples/02_decoding/plot_miyawaki_encoding.py
    :start-after: # beginning/end.
    :end-before: # %%
