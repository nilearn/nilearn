.. _functional_connectomes:

========================================================
Extracting times series to build a functional connectome
========================================================

.. topic:: **Page summary**

   A *functional connectome* is a set of connections representing brain
   interactions between regions. Here we show how to extract activation
   time-series to compute functional connectomes.

.. contents:: **Contents**
    :local:
    :depth: 1


.. topic:: **References**

   * `Varoquaux and Craddock, "Learning and comparing functional
     connectomes across subjects", NeuroImage 2013
     <http://www.sciencedirect.com/science/article/pii/S1053811913003340>`_.

.. _parcellation_time_series:

Time-series from a brain parcellation or "MaxProb" atlas
========================================================

Brain parcellations
-------------------

.. currentmodule:: nilearn.datasets

Regions used to extract the signal can be defined by a "hard"
parcellation. For instance, the :mod:`nilearn.datasets` has functions to
download atlases forming reference parcellation, e.g.,
:func:`fetch_atlas_craddock_2012`, :func:`fetch_atlas_harvard_oxford`,
:func:`fetch_atlas_yeo_2011`.

For instance to retrieve the Harvard-Oxford cortical parcellation, sampled
at 2mm, and with a threshold of a probability of 0.25::

  from nilearn import datasets
  dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
  atlas_filename = dataset.maps
  labels = dataset.labels

Plotting can then be done as::

    from nilearn import plotting
    plotting.plot_roi(atlas_filename)

.. image:: ../auto_examples/01_plotting/images/sphx_glr_plot_atlas_001.png
   :target: ../auto_examples/01_plotting/plot_atlas.html
   :scale: 60

.. seealso::

   * The :ref:`plotting documentation <plotting>`;

   * The :ref:`dataset downloaders <datasets_ref>`.

Extracting signals on a parcellation
------------------------------------

.. currentmodule:: nilearn.input_data

To extract signal on the parcellation, the easiest option is to use the
:class:`nilearn.input_data.NiftiLabelsMasker`. As any "maskers" in
nilearn, it is a processing object that is created by specifying all
the important parameters, but not the data::

    from nilearn.input_data import NiftiLabelsMasker
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

The Nifti data can then be turned to time-series by calling the
:class:`NiftiLabelsMasker` `fit_transform` method, that takes either
filenames or `NiftiImage objects
<http://nipy.org/nibabel/nibabel_images.html>`_::

    time_series = masker.fit_transform(frmi_files, confounds=csv_file)

|

Note that confound signals can be specified in the call. Indeed, to
obtain time series that capture well the functional interactions between
regions, regressing out noise sources is indeed very important
`[Varoquaux & Craddock 2013] <https://hal.inria.fr/hal-00812911/>`_.

.. image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_signal_extraction_001.png
   :target: ../auto_examples/03_connectivity/plot_signal_extraction.html
   :scale: 40
.. image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_signal_extraction_002.png
   :target: ../auto_examples/03_connectivity/plot_signal_extraction.html
   :scale: 40

.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_auto_examples_03_connectivity_plot_signal_extraction.py`.


.. topic:: **Exercise: computing the correlation matrix of rest fmri**
   :class: green

   Try using the information above to compute the correlation matrix of
   the first subject of the ADHD dataset downloaded with
   :func:`nilearn.datasets.fetch_adhd`.

   **Hints:**

   * Inspect the '.keys()' of the object returned by
     :func:`nilearn.datasets.fetch_adhd`.

   * :class:`nilearn.connectome.ConnectivityMeasure` can be used to compute
     a correlation matrix (check the shape of your matrices).

   * :func:`matplotlib.pyplot.imshow` can show a correlation matrix.

   * The example above has the solution.

|

Time-series from a probabilistic atlas
======================================

Probabilistic atlases
---------------------

The definition of regions as by a continuous probability map captures
better our imperfect knowledge of boundaries in brain images (notably
because of inter-subject registration errors). One example of such an
atlas well suited to resting-state data analysis is the `MSDL atlas
<https://team.inria.fr/parietal/18-2/spatial_patterns/spatial-patterns-in-resting-state/>`_
(:func:`nilearn.datasets.fetch_atlas_msdl`).

Probabilistic atlases are represented as a set of continuous maps, in a
4D nifti image. Visualization the atlas thus requires to visualize each
of these maps, which requires accessing them with
:func:`nilearn.image.index_img` (see the :ref:`corresponding example
<sphx_glr_auto_examples_01_plotting_plot_overlay.py>`).

.. image:: ../auto_examples/01_plotting/images/sphx_glr_plot_overlay_001.png
   :target: ../auto_examples/01_plotting/plot_overlay.html
   :scale: 60


Extracting signals from a probabilistic atlas
---------------------------------------------

.. currentmodule:: nilearn.input_data

As with extraction of signals on a parcellation, extracting signals from
a probabilistic atlas can be done with a "masker" object:  the
:class:`nilearn.input_data.NiftiMapsMasker`. It is created by
specifying the important parameters, in particular the atlas::

    from nilearn.input_data import NiftiMapsMasker
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True)

The `fit_transform` method turns filenames or `NiftiImage objects
<http://nipy.org/nibabel/nibabel_images.html>`_ to time series::

    time_series = masker.fit_transform(frmi_files, confounds=csv_file)

The procedure is the same as with `brain parcellations
<parcellation_time_series>`_ but using the :class:`NiftiMapsMasker`, and
the same considerations on using confounds regressors apply.

.. image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_probabilistic_atlas_extraction_001.png
   :target: ../auto_examples/03_connectivity/plot_probabilistic_atlas_extraction.html
   :scale: 30


.. topic:: **Full example**

    A full example of extracting signals on a probabilistic:
    :ref:`sphx_glr_auto_examples_03_connectivity_plot_probabilistic_atlas_extraction.py`.


.. topic:: **Exercise: correlation matrix of rest fMRI on probabilistic atlas**
   :class: green

   Try to compute the correlation matrix of the first subject of the ADHD
   dataset downloaded with :func:`nilearn.datasets.fetch_adhd`
   with the MSDL atlas downloaded via
   :func:`nilearn.datasets.fetch_atlas_msdl`.

   **Hint:** The example above has the solution.


A functional connectome: a graph of interactions
================================================

A square matrix, such as a correlation matrix, can also be seen as a
`"graph" <https://en.wikipedia.org/wiki/Graph_%28mathematics%29>`_: a set
of "nodes", connected by "edges". When these nodes are brain regions, and
the edges capture interactions between them, this graph is a "functional
connectome".

We can display it with the :func:`nilearn.plotting.plot_connectome`
function that take the matrix, and coordinates of the nodes in MNI space.
In the case of the MSDL atlas
(:func:`nilearn.datasets.fetch_atlas_msdl`), the CSV file readily comes
with MNI coordinates for each region (see for instance example:
:ref:`sphx_glr_auto_examples_03_connectivity_plot_probabilistic_atlas_extraction.py`).

..
    For doctesting

    >>> from nilearn import datasets
    >>> atlas_filename = datasets.fetch_atlas_msdl().maps # doctest: +SKIP

For another atlas this information can be computed for each region with
the :func:`nilearn.plotting.find_xyz_cut_coords` function
(see example:
:ref:`sphx_glr_auto_examples_03_connectivity_plot_multi_subject_connectome.py`)::

 >>> from nilearn import image, plotting
 >>> atlas_region_coords = [plotting.find_xyz_cut_coords(img) for img in image.iter_img(atlas_filename)] # doctest: +SKIP



.. image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_probabilistic_atlas_extraction_002.png
   :target: ../auto_examples/03_connectivity/plot_probabilistic_atlas_extraction.html

As you can see, the correlation matrix gives a very "full" graph: every
node is connected to every other one. This is because it also captures
indirect connections. In the next section we will see how to focus on
only direct connections.

|

.. topic:: **References**

  * `Zalesky et al., NeuroImage 2012, "On the use of correlation as a measure of
    network connectivity" <http://www.sciencedirect.com/science/article/pii/S1053811912001784>`_.

  * `Varoquaux et al., NeuroImage 2013, "Learning and comparing functional
    connectomes across subjects" <http://www.sciencedirect.com/science/article/pii/S1053811913003340>`_.
