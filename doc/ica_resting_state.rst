.. _ica_rest:

==================================
ICA of resting-state fMRI datasets
==================================

Preprocessing
==============

Loading
-------

As seen in :ref:`previous sections <downloading_data>`, we fetch the data from
internet and load it with a provided function:


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


