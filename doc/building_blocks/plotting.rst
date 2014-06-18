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


Different display modes
========================

.. |plot_ortho| image:: ../auto_examples/images/plot_demo_plotting_5.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_z_many| image:: ../auto_examples/images/plot_demo_plotting_6.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_x| image:: ../auto_examples/images/plot_demo_plotting_7.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_x_small| image:: ../auto_examples/images/plot_demo_plotting_8.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_z_small| image:: ../auto_examples/images/plot_demo_plotting_9.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_xz| image:: ../auto_examples/images/plot_demo_plotting_10.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_yx| image:: ../auto_examples/images/plot_demo_plotting_11.png
     :target: ../auto_examples/plot_demo_plotting.html
     :scale: 50

.. |plot_yz| image:: ../auto_examples/images/plot_demo_plotting_12.png
     :target: ../auto_examples/plot_demo_plotting.html
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

