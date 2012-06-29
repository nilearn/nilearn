.. _searchlight:

===========================================================
Searchlight : finding voxels containing maximum information
===========================================================

.. currentmodule:: nisl.searchlight

Searchlight principle
=====================

Searchlight was introduced in `Information-based functional brain mapping
<http://www.pnas.org/content/103/10/3863>`_, Nikolaus Kriegeskorte,
Rainer Goebel and Peter Bandettini (PNAS 2006) and consists in scanning the
images volume with a *searchlight*. Briefly, a ball of given radius is
scanned across the brain volume and the prediction accuracy of a
classifier trained on the corresponding voxels is measured.

Preprocessing
=============

Loading
-------

As seen in previous tutorial, this is rather easy thanks to *nisl* dataset
manager.

.. literalinclude:: ../plot_haxby_searchlight.py
    :start-after: ### Load Haxby dataset ########################################################
    :end-before: ### Preprocess data ###########################################################

Preparing data
--------------

For this tutorial we need:

- to put X in the form *n_samples* x *n_features*
- compute a mean image for visualisation background
- detrend the data

.. literalinclude:: ../plot_haxby_searchlight.py
    :start-after: ### Preprocess data ###########################################################
    :end-before: ### Prepare the masks #########################################################

Masking
-------

One of the main element that distinguich Searchlight from other algorithms is
this notion of structuring element that scan the entire volume. If this seems
rather intuitive, it has in fact an impact on the masking procedure.

Most of the time, fMRI data is masked and then given to the algorithm. This is
not possible in the case of Searchlight because, to compute the score of
non-masked voxels, some masked voxels may be needed. This is why two masks will
be used here :

- *mask* is the anatomical mask
- *process_mask* is a subset of mask and contains voxels to be processed.

*process_mask* will then be used to restrain computation to one slice, in the
back of the brain. *mask* will ensure that no value outside of the brain is
taken into account when iterating with the sphere.

.. literalinclude:: ../plot_haxby_searchlight.py
        :start-after: #   up computation)
        :end-before: ### Restrict to faces and houses ##############################################

Restricting the dataset
-----------------------

Like *haxby_decoding* example, we limit our analysis to the `face` and `house` conditions:

.. literalinclude:: ../plot_haxby_searchlight.py
    :start-after: ### Restrict to faces and houses ##############################################
    :end-before: ### Searchlight ###############################################################
	
Third Step : Set up the cross validation
========================================

Searchlight will iterate on the volume and give a score to each voxel. This
score is computed by running a classifier on selected voxels. In order to make
this score as accurate as possible (and avoid overfitting), a cross validation
is made.

Classifier
----------

The classifier used by default by Searchlight is LinearSVC with C=1 but this
can be customed easily by passing an estimator parameter to the cross
validation.

See scikit-learn documentation for
`other classifiers <http://scikit-learn.org/supervised_learning.html>`_

Score function
--------------

Here we use precision as metrics to measures proportion of true
positives among all positives results for one class.

Many others are available in `scikit-learn documentation
<http://scikit-learn.org/supervised_learning.html>`_

.. literalinclude:: ../plot_haxby_searchlight.py
    :start-after: # all positives results for one class.
    :end-before: ### Define the cross-validation scheme used for validation.

Cross validation
----------------

As Searchlight is a little costly, we have chosen a cross validation method
that do not take too much time. *K*-Fold along with *K* = 4 is a good compromise
between running time and result.

.. literalinclude:: ../plot_haxby_searchlight.py
    :start-after: # set once and the others as learning sets
    :end-before: ### Fit #######################################################################

Running Searchlight
===================

Running Searchlight is straightforward now that everything is set. The only
parameter left is the radius of the ball that will run through the data.
Kriegskorte uses a 4mm radius because it yielded the best detection
performance in his simulation.

.. literalinclude:: ../plot_haxby_searchlight.py
    :start-after: ### Fit #######################################################################
    :end-before: ### Visualization #############################################################
	
Visualisation
=============

Searchlight
-----------

As the activation map is cropped, we use the mean image of all scans as a
background. We can see here that voxels in the visual cortex contains
information to distinguish pictures showed to the volunteer, which was the
expected result.

.. figure:: auto_examples/images/plot_haxby_searchlight_1.png
   :target: auto_examples/plot_haxby_searchlight.html
   :align: center
   :scale: 60

.. literalinclude:: ../plot_haxby_searchlight.py
    :start-after: ### Visualization #############################################################
    :end-before: ### Show the F_score

F_score
-------

Another commonly used algorithm to find salient voxel is the ANOVA (analysis of
variance). Here we use is to compute the *p_values* of the voxels. The
*p_value* is the probability of getting the observed values assuming that
nothing happens (i.e. the null hypothesis is true). Therefore, a small
*p-value* indicates that there is a small chance of getting this data if no
real difference existed, so the observed voxel must be significant.

As the policy "the smaller, the better" is not very intuitive, we use the log
and negate the result to obtain a more comprehensive map.

.. figure:: auto_examples/images/plot_haxby_searchlight_2.png
   :target: auto_examples/plot_haxby_searchlight.html
   :align: center
   :scale: 60

.. literalinclude:: ../plot_haxby_searchlight.py
    :start-after: ### Show the F_score
