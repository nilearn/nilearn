Nilearn
=======

.. container:: index-paragraph

    Nilearn enables **approachable and versatile analyses of brain
    volumes**. It provides statistical and machine-learning tools, with
    **instructive documentation & open community**.

    It supports general linear model (GLM) based analysis and leverages
    the :sklearn:`scikit-learn <>` Python toolbox
    for multivariate statistics with applications such as predictive modelling,
    classification, decoding, or connectivity analysis.


.. grid::

    .. grid-item-card:: :fas:`rocket` Quickstart
        :link: quickstart
        :link-type: ref
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Get started with Nilearn

    .. grid-item-card:: :fas:`th` Examples
        :link: auto_examples/index.html
        :link-type: url
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Discover functionalities by reading examples

    .. grid-item-card:: :fas:`book` User guide
        :link: user_guide
        :link-type: ref
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Learn about neuroimaging analysis

Featured examples
-----------------

.. grid::

  .. grid-item-card::
    :link: auto_examples/01_plotting/plot_demo_glass_brain.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/01_plotting/images/sphx_glr_plot_demo_glass_brain_002.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Glass brain plotting

        Explore how to retrieve data and plot whole brain cuts
        in glass mode.

  .. grid-item-card::
    :link: auto_examples/03_connectivity/plot_inverse_covariance_connectome.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/03_connectivity/images/sphx_glr_plot_inverse_covariance_connectome_004.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Computing a connectome with sparse inverse covariance

        Construct a functional connectome using the sparse inverse covariance,
        and display the corresponding graph and matrix.

  .. grid-item-card::
    :link: auto_examples/01_plotting/plot_3d_map_to_surface_projection.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/01_plotting/images/sphx_glr_plot_3d_map_to_surface_projection_001.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Making a surface plot of a 3D statistical map

        Project a 3D statistical map onto a cortical mesh
        and display the surface map as png or in interactive mode.

  .. grid-item-card::
    :link: auto_examples/00_tutorials/plot_decoding_tutorial.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/00_tutorials/images/sphx_glr_plot_decoding_tutorial_001.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Introduction tutorial to fMRI decoding

        Learn to perform decoding with nilearn. Reproduce the Haxby 2001 study on a face vs cat discrimination task in a mask of the ventral stream.

  .. grid-item-card::
    :link: auto_examples/02_decoding/plot_oasis_vbm.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/02_decoding/images/sphx_glr_plot_oasis_vbm_002.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Voxel-Based Morphometry on Oasis dataset

        Study the relationship between aging and gray matter density
        using data from the OASIS project.

  .. grid-item-card::
    :link: auto_examples/03_connectivity/plot_data_driven_parcellations.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/03_connectivity/images/sphx_glr_plot_data_driven_parcellations_001.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Clustering methods to learn a brain parcellation from fMRI

        Use spatially-constrained Ward-clustering, KMeans, Hierarchical KMeans
        and Recursive Neighbor Agglomeration (ReNA) to create a set of parcels,
        and display them.

  .. grid-item-card::
    :link: auto_examples/03_connectivity/plot_compare_decomposition.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_001.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Deriving spatial maps from group fMRI data using ICA and Dictionary Learning

        Derive spatial maps or networks from group fMRI data
        using two popular decomposition methods, ICA and Dictionary learning
        on data of children and young adults watching movies.

  .. grid-item-card::
    :link: auto_examples/02_decoding/plot_haxby_frem.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/02_decoding/images/sphx_glr_plot_haxby_frem_001.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Decoding with FREM: face vs house object recognition

        Use fast ensembling of regularized models (FREM)
        to decode a face vs house discrimination task from Haxby 2001 study.

  .. grid-item-card::
    :link: auto_examples/02_decoding/plot_haxby_searchlight.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/02_decoding/images/sphx_glr_plot_haxby_searchlight_001.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Searchlight analysis of face vs house recognition

        Fit a classifier a large amount of times
        in order to distinguish between face- and house-related
        cortical areas.


.. toctree::
   :hidden:
   :includehidden:
   :titlesonly:

   quickstart.md
   auto_examples/index.rst
   user_guide.rst
   modules/index.rst
   glossary.rst

.. toctree::
   :hidden:
   :caption: Development

   development.rst
   maintenance.rst
   changes/whats_new.rst
   authors.rst
   GitHub Repository <https://github.com/nilearn/nilearn>


Nilearn is part of the :nipy:`NiPy ecosystem <>`.
