.. _gpu_usage:

=============================================================
Leverage GPUs for your neuroimaging machine-learning pipeline
=============================================================

Nilearn is dependent on scikit-learn for its machine learning tools.
With the recent developments in scikit-learn, there is now
:sklearn:`a growing list of estimators <modules/array_api.html#support-for-array-api-compatible-inputs>`
that support GPU acceleration via the `Array API <https://data-apis.org/array-api>`_.

This means that you can now also use GPUs to speed up your neuroimaging
machine-learning pipelines in Nilearn.

We will demonstrate this speedup using the :ref:`sphx_glr_auto_examples_02_decoding_plot_miyawaki_encoding.py` example,
which implements an encoding model using
:class:`sklearn.linear_model.Ridge` to predict fMRI data
from visual stimuli.

Initial setup to access GPU acceleration
========================================

You would need to have a compatible GPU and scikit-learn >= 1.8 to run the
code here:

.. code-block:: bash

    pip install -U scikit-learn>=1.8

In addition, you need to either `install CuPy
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


Now we can proceed to load and prepare the data just as in the original
example: :ref:`sphx_glr_auto_examples_02_decoding_plot_miyawaki_encoding.py`.

Time the original CPU version of the code
=========================================

The first important difference from the original example is that, at the time
of writing, the GPU-acceleration in
:class:`sklearn.linear_model.Ridge` is only available
when the ``solver`` parameter is set to ``"svd"`` as mentioned in the
:sklearn:`scikit-learn documentation. <modules/array_api.html#support-for-array-api-compatible-inputs>`

So we need to initialize the estimator as follows:

.. code-block:: python

    from sklearn.linear_model import Ridge

    estimator = Ridge(alpha=100.0, solver="svd")

Now we can run a grid search over a range of ``alpha`` values and use
IPython magic to time the execution like this:

.. code-block:: python

    %%time
    from sklearn.model_selection import GridSearchCV

    param_grid = dict(alpha=np.logspace(-6, 6, 100))

    search_cv_cpu = GridSearchCV(estimator, param_grid)
    search_cv_cpu.fit(stimuli, fmri_data)

    # CPU times: user 3min 44s, sys: 28.2 s, total: 4min 13s
    # Wall time: 2min 31s

So we see that the original CPU version of the code takes around 2 minutes and
30 seconds to run.

Move to GPU acceleration
========================

Now let's run the same code with GPU acceleration. For that, we first need to
convert our data to CuPy arrays:

.. code-block:: python

    import cupy as cp

    fmri_data_cupy = cp.asarray(fmri_data)
    stimuli_cupy = cp.asarray(stimuli)


Now using the CuPy arrays, we can run the same grid search as before:

.. code-block:: python

    %%time
    search_cv_gpu = GridSearchCV(estimator, param_grid)
    search_cv_gpu.fit(stimuli_cupy, fmri_data_cupy)

    # CPU times: user 25.1 s, sys: 879 ms, total: 26 s
    # Wall time: 26 s

Just in case you are using PyTorch instead of CuPy, you can convert the data to
PyTorch tensors like this:

.. code-block:: python

    import torch

    fmri_data_torch = torch.from_numpy(fmri_data)
    stimuli_torch = torch.from_numpy(stimuli)

And then run the same grid search as before:

.. code-block:: python

    %%time
    search_cv_gpu = GridSearchCV(estimator, param_grid)
    search_cv_gpu.fit(stimuli_torch, fmri_data_torch)

    # CPU times: user 38.2 s, sys: 46.5 ms, total: 38.3 s
    # Wall time: 38.5 s

So overall we see about a 70-80% reduction in execution time when using GPU
acceleration in this case. However note that the exact speedup you get may
depend on the specific GPU you have, the size of your data and the kind of
analysis you are doing.
