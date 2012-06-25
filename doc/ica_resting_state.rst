.. _ica_rest:

==================================
ICA of resting-state fMRI datasets
==================================

Preprocessing
==============

Loading
-------

Thanks to *nisl* dataset manager, fetching the dataset is rather easy. Do not
forget to set your environment variable *NISL_DATA* if you want your dataset
to be stored in a specific path.

.. literalinclude:: ../plot_ica_resting_state.py
    :start-after: ### Load nyu_rest dataset #####################################################
    :end-before: ### Preprocess ################################################################

Conctenating, smoothing and masking
------------------------------------

.. literalinclude:: ../plot_ica_resting_state.py
    :start-after: ### Preprocess ################################################################
    :end-before: ### Apply ICA #################################################################

Applying ICA
==============

.. literalinclude:: ../plot_ica_resting_state.py
    :start-after: ### Apply ICA #################################################################
    :end-before: ### Visualize the results #####################################################

Visualizing the results
========================

Visualization follows similarly as in the previous examples. Remember
that we use masked arrays (`np.ma`) to create transparency in the
overlays.

.. literalinclude:: ../plot_ica_resting_state.py
    :start-after: ### Visualize the results #####################################################

.. |left_img| image:: auto_examples/images/plot_ica_resting_state_1.png
   :target: auto_examples/plot_ica_resting_state.html
   :width: 49%

.. |right_img| image:: auto_examples/images/plot_ica_resting_state_2.png
   :target: auto_examples/plot_ica_resting_state.html
   :width: 49%

|left_img| |right_img|


