.. _decoding_intro:


=============================
An introduction to decoding
=============================

This section gives an introduction to the main concept of decoding:
predicting from brain images.

The discussion and examples are articulated on the analysis of the Haxby
2001 dataset, showing how to predict from :term:`fMRI` images the stimuli that
the subject is viewing. However the process is the same in other settings
predicting from other brain imaging modalities, for instance predicting
phenotype or diagnostic status from :term:`VBM` (Voxel Based Morphometry) maps
(as illustrated in :ref:`a more complex example
<sphx_glr_auto_examples_02_decoding_plot_oasis_vbm.py>`), or from FA maps
to capture diffusion mapping.

.. note::
  This documentation only aims at explaining the necessary concepts and common
  pitfalls of decoding analysis. For an introduction on the code to use please
  refer to : :ref:`sphx_glr_auto_examples_00_tutorials_plot_decoding_tutorial.py`


Loading and preparing the data
===============================

The Haxby 2001 experiment
-------------------------

In the Haxby experiment, subjects were presented visual stimuli from
different categories. We are going to predict which category the subject is
seeing from the :term:`fMRI` activity recorded in regions of the ventral visual system.
Significant prediction shows that the signal in the region contains
information on the corresponding category.

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_stimuli_007.png
   :target: ../auto_examples/02_decoding/plot_haxby_stimuli.html
   :scale: 30
   :align: center

   Face stimuli

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_stimuli_004.png
   :target: ../auto_examples/02_decoding/plot_haxby_stimuli.html
   :scale: 30
   :align: center

   Cat stimuli

.. figure:: ../auto_examples/01_plotting/images/sphx_glr_plot_haxby_masks_001.png
   :target: ../auto_examples/01_plotting/plot_haxby_masks.html
   :scale: 30
   :align: center

   Masks

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_full_analysis_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_full_analysis.html
   :scale: 35
   :align: center

   Decoding scores per mask

_____

.. topic:: **fMRI: using beta maps of a first-level analysis**

   The Haxby experiment is unusual because the experimental paradigm is
   made of many blocks of continuous stimulation. Most cognitive
   experiments have a more complex temporal structure with rich sequences
   of events
   (:ref:`more on data input <loading_data>`).

   The standard approach to decoding consists in fitting a first-level
   :ref:`general linear model (or GLM) <glm_intro>` to retrieve one response
   map (a beta map) per trial as shown in
   :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_glm_decoding.py`.
   This is sometimes known as "beta-series regressions" (see :footcite:t:`Mumford2012`).
   These maps can then be input to the decoder as below,
   predicting the conditions associated to trial.

   For simplicity, we will work on the raw time-series of the data.
   However, **it is strongly recommended that you fit a first-level model to
   include an hemodynamic response function (HRF) model and isolate the
   responses from various confounds** as demonstrated in :ref:`a more advanced example <sphx_glr_auto_examples_02_decoding_plot_haxby_glm_decoding.py>`.


Loading the data into nilearn
-----------------------------

.. topic:: **Full code example**

   The documentation here just gives the big idea. A full code example,
   with explanation, can be found on
   :ref:`sphx_glr_auto_examples_00_tutorials_plot_decoding_tutorial.py`

* **Starting an environment**: Launch IPython via "ipython --matplotlib"
  in a terminal, or use the Jupyter notebook.

* **Retrieving the data**: In the tutorial, we load the data using nilearn
  data downloading function, :func:`nilearn.datasets.fetch_haxby`.
  However, all this function does is to download the data and return
  paths to the files downloaded on the disk. To input your own data to
  nilearn, you can pass in the path to your own files
  (:ref:`more on data input <loading_data>`).


* **Masking fMRI data**: To perform the analysis on some :term:`voxels<voxel>` only, we will
  provide a spatial mask of :term:`voxels<voxel>` to keep, which is provided with the dataset
  (here ``mask_vt`` a mask of the ventral temporal cortex that comes with data).

* **Loading the behavioral labels**: Behavioral information is often stored
  in a text file such as a CSV, and must be load with
  **numpy.genfromtxt** or `pandas <https://pandas.pydata.org/>`_

* **Sample mask**: Masking some of the time points
  may be useful to
  restrict to a specific pair of conditions (*eg* cats versus faces).

.. seealso::
  * :ref:`masking`
    To better control this process of spatial masking and add additional signal
    processing steps (smoothing, filtering, standardizing...), we could
    explicitly define a masker :  :class:`nilearn.maskers.NiftiMasker`.
    This object extracts :term:`voxels<voxel>` belonging to a given spatial mask and converts
    their signal to a 2D data matrix with a shape (n_timepoints, n_voxels)
    (see :ref:`mask_4d_2_3d` for a discussion on using masks).

.. note::
  Seemingly minor data preparation can matter a lot on the final score,
  for instance standardizing the data.


Performing a simple decoding analysis
=======================================

A few definitions
---------------------

When doing predictive analysis you train an estimator to predict a variable of
interest to you. Or in other words to predict a condition label **y** given a
set **X** of imaging data.

This is always done in at least two steps:

* first a ``fit`` during which we "learn" the parameters of the model that make
  good predictions. This is done on some "training data" or "training set".
* then a ``predict`` step where the "fitted" model is used to make prediction
  on new data. Here, we just have to give the new set of images (as the target
  should be unknown). These are called "test data" or "test set".

All objects used to make prediction in Nilearn will at least have functions for
these steps : a ``fit`` function and a ``predict`` function.

.. warning::

    **Do not predict on data used by the fit: this would yield misleadingly
    optimistic scores.**


A first estimator
-----------------

To perform decoding, we need a model that can learn some relations
between **X** (the imaging data) and **y** the condition label. As a default,
Nilearn uses `Support Vector Classifier
<https://scikit-learn.org/stable/modules/svm.html>`_ (or SVC) with a
linear kernel. This is a simple yet performant choice that works in a wide
variety of problems.

.. seealso::

   `The scikit-learn documentation on SVMs
   <https://scikit-learn.org/stable/modules/svm.html>`_

Decoding made easy
-------------------

Nilearn makes it easy to train a model with a principled pipeline using the
:class:`nilearn.decoding.Decoder` object. Using the mask we defined before
and an SVC estimator as we already introduced, we can create a pipeline in
two lines. The additional ``standardize=True`` argument adds a normalization
of images signal to a zero mean and unit variance, which will improve
performance of most estimators.

.. code-block:: python

     from nilearn.decoding import Decoder
     decoder = Decoder(estimator='svc', mask=mask_filename)

Then we can fit it on the images and the conditions we chose before.

.. code-block:: python

     decoder.fit(fmri_niimgs, conditions)

This decoder can now be used to predict conditions for new images !
Be careful though, as we warned you, predicting on images that were used to
``fit`` your model should never be done.


Measuring prediction performance
--------------------------------

One of the most common interests of decoding is to measure how well we can learn
to predict various targets from our images to have a sense of which information
is really contained in a given region of the brain. To do this, we need ways to
measure the errors we make when we do prediction.


Cross-validation
................

We cannot measure prediction error on the same set of data that we have
used to fit the estimator: it would be much easier than on new data, and
the result would be meaningless. We need to use a technique called
*cross-validation* to split the data into different sets, we can then ``fit`` our
estimator on some set and measure an unbiased error on another set.

The easiest way to do cross-validation is the `K-Fold strategy
<https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`_.
If you do 5-fold cross-validation manually, you split your data in 5 folds,
use 4 folds to ``fit`` your estimator, and 1 to ``predict`` and measure the errors
made by your estimators. You repeat this for every combination of folds, and get
5 prediction "scores", one for each fold.

During the ``fit``, :class:`nilearn.decoding.Decoder` object implicitly used a
cross-validation: Stratified K-fold by default. You can easily inspect
the prediction "score" it got in each fold.

.. code-block:: python

     print(decoder.cv_scores_)


Choosing a good cross-validation strategy
.........................................

There are many cross-validation strategies possible, including K-Fold or
leave-one-out. When choosing a strategy, keep in mind that the test set should
be as little correlated as possible with the train set and have enough samples
to enable a good measure the prediction error (at least 10-20% of the data as a
rule of thumb).

As a general advice :

* To train a decoder on one subject data, try to leave at least one run
  out to have an independent test.

* To train a decoder across different subject data, leaving some subjects data
  out is often a good option.

* In any case leaving only one image as test set (leave-one-out) is often
  the worst option (see :footcite:t:`Varoquaux2017`).


To improve our first pipeline for the Haxby example, we can leave one entire
run out. To do this, we can pass a ``LeaveOneGroupOut`` cross-validation
object from scikit-learn to our ``Decoder``. Fitting it with the information of
groups=`run_labels` will use one run as test set.

.. note::
  Full code example can be found at :
  :ref:`sphx_glr_auto_examples_00_tutorials_plot_decoding_tutorial.py`


Choice of the prediction accuracy measure
.........................................

Once you have a prediction about new data and its real label (the *ground truth*)
there are different ways to measure a *score* that summarizes its performance.

The default metric used for measuring errors is the accuracy score, i.e.
the number of total errors. It is not always a sensible metric,
especially in the case of very imbalanced classes, as in such situations
choosing the dominant class can achieve a low number of errors.

Other metrics, such as the :term:`AUC` (Area Under the Curve, for the
:term:`ROC`: the Receiver Operating Characteristic), can be used through the
``scoring`` argument of :class:`nilearn.decoding.Decoder`.

.. seealso::
  the `list of scoring options
  <https://scikit-learn.org/stable/modules/model_evaluation.html>`_

Prediction accuracy at chance using simple strategies
.....................................................

When performing decoding, prediction performance of a model can be checked against
null distributions or random predictions. For this, we guess a chance level score using
simple strategies while predicting condition **y** with **X** imaging data.

In Nilearn, we wrap
`Dummy estimators <https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators>`_
into the :class:`nilearn.decoding.Decoder` that
can be readily used to estimate this chance level score with the same model parameters
that was previously used for real predictions. This allows us to compare whether the
model is better than chance or not.

.. topic:: **Putting it all together**

    The :ref:`ROI-based decoding example
    <sphx_glr_auto_examples_02_decoding_plot_haxby_full_analysis.py>`
    does a decoding analysis per mask, giving the f1-score and chance score of
    the prediction for each object.

    It uses all the notions presented above, with ``for`` loop to iterate
    over masks and categories and Python dictionaries to store the
    scores.


.. figure:: ../auto_examples/01_plotting/images/sphx_glr_plot_haxby_masks_001.png
   :target: ../auto_examples/01_plotting/plot_haxby_masks.html
   :scale: 55
   :align: center

   Masks


.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_full_analysis_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_full_analysis.html
   :scale: 70
   :align: center


Visualizing the decoder's weights
---------------------------------

During ``fit`` step, the :class:`nilearn.decoding.Decoder` object retains the
coefficients of best models for each class in ``decoder.coef_img_``.


.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_anova_svm_001.png
   :target: ../auto_examples/plot_decoding_tutorial.html
   :align: center
   :scale: 65

.. note::
  Full code for the above can be found on
  :ref:`sphx_glr_auto_examples_00_tutorials_plot_decoding_tutorial.py`


.. seealso::
  * :ref:`plotting`


Decoding without a mask: Anova-SVM
==================================

Dimension reduction with feature selection
------------------------------------------

If we do not start from a mask of the relevant regions, there is a very
large number of voxels and not all are useful for
face vs cat prediction. We thus add a `feature selection
<https://scikit-learn.org/stable/modules/feature_selection.html>`_
procedure. The idea is to select the ``k`` voxels most correlated to the
task through a simple F-score based feature selection (a.k.a.
`Anova <https://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_)

You can directly choose to keep only a certain percentage of voxels in the
:class:`nilearn.decoding.Decoder` object through the ``screening_percentile``
argument. To keep the 10% most correlated voxels, just create us this parameter :

.. literalinclude:: ../../examples/02_decoding/plot_haxby_anova_svm.py
   :start-after: # on nested cross-validation.
   :end-before: # Visualize the results

.. note::

    Providing a region-of-interest mask may interact with
    the ``screening_percentile`` parameter, particularly
    in cases where the mask extent is small relative to the
    total brain volume. In these cases, there may not be
    enough features in the mask to allow for further
    sub-selection with ``screening_percentile``.

Visualizing the results
-----------------------

To visualize the results, :class:`nilearn.decoding.Decoder` handles two main steps for you :

* first get the support vectors of the SVC and inverse the feature selection mechanism
* then, inverse the masking process to link weights to their spatial
  position and plot

.. literalinclude:: ../../examples/02_decoding/plot_haxby_anova_svm.py
   :start-after: # Visualize the results
   :end-before: # Saving the results as a Nifti file may also be important

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_anova_svm_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_anova_svm.html
   :align: center
   :scale: 65

.. seealso::
  * :ref:`plotting`


.. topic:: **Final script**

    The complete script to do an SVM-Anova analysis can be found as
    :ref:`an example <sphx_glr_auto_examples_02_decoding_plot_haxby_anova_svm.py>`.

.. seealso::
  * :ref:`frem`
  * :ref:`space_net`
  * :ref:`searchlight`


References
----------

.. footbibliography::
