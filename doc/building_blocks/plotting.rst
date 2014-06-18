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

.. |plot_stat_map| image:: ../auto_examples/images/plot_demo_plotting_1.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_anat| image:: ../auto_examples/images/plot_demo_plotting_2.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_roi| image:: ../auto_examples/images/plot_demo_plotting_3.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_epi| image:: ../auto_examples/images/plot_demo_plotting_4.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. A temporary hack to avoid a sphinx bug
.. |hack| raw:: html

   <br>

================= =========================================================
================= =========================================================
|plot_anat|        :func:`plot_anat`
                   |hack|
                   Plotting an anatomical image

|plot_epi|        :func:`plot_epi`
                  |hack|
                  Plotting an EPI, or T2* image

|plot_stat_map|   :func:`plot_stat_map`
                  |hack|
                  Plotting a statistical map, like a T-map, a Z-map, or
                  an ICA, with an optional background

|plot_roi|        :func:`plot_roi`
                  |hack|
                  Plotting ROIs, or a mask, with an optional background

**plot_img**      :func:`plot_img`
                  |hack|
                  General-purpose function, with no specific presets
================= =========================================================



