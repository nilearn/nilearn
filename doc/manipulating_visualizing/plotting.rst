.. _plotting:

======================
Plotting brain images
======================

Nilearn comes with plotting function to display brain maps coming from
Nifti-like images, in the :mod:`nilearn.plotting` module.

.. currentmodule:: nilearn.plotting

Different plotting functions
=============================

Nilearn has a set of plotting functions to plot brain volumes that are
fined tuned to specific applications. Amongst other things, they use
different heuristics to find cutting coordinates.

.. |plot_stat_map| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_plotting_001.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_plotting.html
     :scale: 50

.. |plot_glass_brain| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_glass_brain_extensive_001.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_glass_brain_extensive.html
     :scale: 50

.. |plot_connectome| image:: ../auto_examples/connectivity/images/sphx_glr_plot_inverse_covariance_connectome_002.png
     :target: ../auto_examples/connectivity/plot_inverse_covariance_connectome.html
     :scale: 50

.. |plot_anat| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_plotting_003.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_plotting.html
     :scale: 50

.. |plot_roi| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_plotting_004.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_plotting.html
     :scale: 50

.. |plot_epi| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_plotting_005.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_plotting.html
     :scale: 50

.. |plot_prob_atlas| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_prob_atlas_003.png
     :target: ../auto_examples/manipulating_visualizing/plot_prob_atlas.html
     :scale: 50

.. A temporary hack to avoid a sphinx bug
.. |hack| raw:: html

   <br>


=================== =========================================================
=================== =========================================================
|plot_anat|          :func:`plot_anat`
                     |hack|
                     Plotting an anatomical image

|plot_epi|           :func:`plot_epi`
                     |hack|
                     Plotting an EPI, or T2* image

|plot_glass_brain|   :func:`plot_glass_brain`
                     |hack|
                     Glass brain visualization. By default plots maximum
                     intensity projection of the absolute values. To plot
                     positive and negative values set plot_abs parameter to
                     False.

|plot_stat_map|      :func:`plot_stat_map`
                     |hack|
                     Plotting a statistical map, like a T-map, a Z-map, or
                     an ICA, with an optional background

|plot_roi|           :func:`plot_roi`
                     |hack|
                     Plotting ROIs, or a mask, with an optional background

|plot_connectome|    :func:`plot_connectome`
                     |hack|
                     Plotting a connectome

|plot_prob_atlas|    :func:`plot_prob_atlas`
                     |hack|
                     Plotting 4D probabilistic atlas maps

**plot_img**         :func:`plot_img`
                     |hack|
                     General-purpose function, with no specific presets
=================== =========================================================


.. warning:: **Opening too many figures without closing**

   Each call to a plotting function creates a new figure by default. When
   used in non-interactive settings, such as a script or a program, these
   are not displayed, but still accumulate and eventually lead to slowing
   the execution and running out of memory.

   To avoid this, you must close the plot as follow::

    >>> from nilearn import plotting
    >>> display = plotting.plot_stat_map(img)     # doctest: +SKIP
    >>> display.close()     # doctest: +SKIP

.. seealso::

   :ref:`sphx_glr_auto_examples_manipulating_visualizing_plot_dim_plotting.py`

Different display modes
========================

.. |plot_ortho| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_001.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 50

.. |plot_z_many| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_002.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 30

.. |plot_x| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_003.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 50

.. |plot_x_small| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_004.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 50

.. |plot_z_small| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_005.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 50

.. |plot_xz| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_006.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 50

.. |plot_yx| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_007.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 50

.. |plot_yz| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_008.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 50


================= =========================================================
================= =========================================================
|plot_ortho|       `display_mode='ortho', cut_coords=(36, -27, 60)` 
                   |hack|
                   Ortho slicer: 3 cuts along the x, y, z directions

|plot_z_many|      `display_mode='z', cut_coords=5`
                   |hack|
                   Cutting in the z direction, specifying the number of
                   cuts

|plot_x|           `display_mode='x', cut_coords=(-36, 36)`
                   |hack|
                   Cutting in the x direction, specifying the exact
                   cuts

|plot_x_small|     `display_mode='x', cut_coords=1`
                   |hack|
                   Cutting in the x direction, with only 1 cut, that is
                   automatically positionned

|plot_z_small|     `display_mode='z', cut_coords=1, colorbar=False`
                   |hack|
                   Cutting in the z direction, with only 1 cut, that is
                   automatically positionned

|plot_xz|          `display_mode='xz', cut_coords=(36, 60)`
                   |hack|
                   Cutting in the x and z direction, with cuts manually
                   positionned

|plot_yx|          `display_mode='yx', cut_coords=(-27, 36)`
                   |hack|
                   Cutting in the y and x direction, with cuts manually
                   positionned

|plot_yz|          `display_mode='yz', cut_coords=(-27, 60)`
                   |hack|
                   Cutting in the y and z direction, with cuts manually
                   positionned


================= =========================================================

Adding overlays, edges and contours
====================================

To add overlays, contours, or edges, use the return value of the plotting
functions. Indeed, these return a display object, such as the
:class:`nilearn.plotting.displays.OrthoSlicer`. This object represents the
plot, and has methods to add overlays, contours or edge maps::

        display = plotting.plot_epi(...)

.. |plot_edges| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_009.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 50

.. |plot_contours| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_demo_more_plotting_010.png
     :target: ../auto_examples/manipulating_visualizing/plot_demo_more_plotting.html
     :scale: 50

================= =========================================================
================= =========================================================
|plot_edges|       `display.add_edges(img)`
                   |hack|
                   Add a plot of the edges of `img`, where edges are
                   extracted using a Canny edge-detection routine. This
                   is typically useful to check registration. Note that
                   `img` should have some visible sharp edges. Typically
                   an EPI img does not, but a T1 does.

|plot_contours|    `display.add_contours(img, levels=[.5], colors='r')`
                   |hack|
                   Add a plot of the contours of `img`, where contours
                   are computed for constant values, specified in
                   'levels'. This is typically useful to outline a mask,
                   or ROI on top of another map.
                   |hack|
                   **Example:** :ref:`sphx_glr_auto_examples_manipulating_visualizing_plot_haxby_masks.py`


**add_overlay**   `display.add_overlay(img, cmap=plotting.cm.purple_green, threshold=3)`
                  |hack|
                  Add a new overlay on the existing figure
                  |hack|
                  **Example:** :ref:`sphx_glr_auto_examples_manipulating_visualizing_plot_overlay.py`


================= =========================================================

Displaying or saving to an image file
=====================================

To display the figure when running a script, you need to call
:func:`nilearn.plotting.show`: (this is just an alias to
:func:`matplotlib.pyplot.show`)::

    >>> from nilearn import plotting
    >>> plotting.show() # doctest: +SKIP

The simplest way to output an image file from the plotting functions is
to specify the `output_file` argument::

    >>> from nilearn import plotting
    >>> plotting.plot_stat_map(img, output_file='pretty_brain.png')     # doctest: +SKIP

In this case, the display is closed automatically and the plotting
function returns None.

|

The display object returned by the plotting function has a savefig method
that can be used to save the plot to an image file::

    >>> from nilearn import plotting
    >>> display = plotting.plot_stat_map(img)     # doctest: +SKIP
    >>> display.savefig('pretty_brain.png')     # doctest: +SKIP
    # Don't forget to close the display
    >>> display.close()     # doctest: +SKIP


Some tips for better rendering
==============================

A typical use of the plotting function is to display some functional
characteristics (statistical parametric map, independent component),
while having an anatomical image as background, that implicitly defines
the location of all the features seen in the functional image.

By default, the background image is the MNI152 template (T1)
image. However, the rendering with an actual anatomical image is
generally better.

Using the default setting, the anatomical image often seem overly
dark. This is easily fixed by reducing the dimming parameter,
e.g. to 0.

.. |plot_mni_background| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_spms_001.png
     :target: ../auto_examples/manipulating_visualizing/plot_spms.html
     :scale: 50

.. |plot_anat_background| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_spms_002.png
     :target: ../auto_examples/manipulating_visualizing/plot_spms.html
     :scale: 50

.. |plot_bright_background| image:: ../auto_examples/manipulating_visualizing/images/sphx_glr_plot_spms_003.png
     :target: ../auto_examples/manipulating_visualizing/plot_spms.html
     :scale: 50

================= =========================================================
================= =========================================================

|plot_mni_background|    `activation image on MNI template` 
                              |hack|
			      The background image is the T1 MNI template.

|plot_anat_background|    `activation image on subject anatomical image` 
                              |hack|
			      The background image is the subject anatomy.

|plot_bright_background|    `activation image on brighted subject anatomical image` 
                              |hack|
			      The background image is the subject a,atomy --brighter.

