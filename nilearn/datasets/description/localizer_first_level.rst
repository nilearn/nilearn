.. _localizer_first_level:

localizer first level dataset
=============================

Access
------
See :func:`nilearn.datasets.fetch_localizer_first_level`.

Notes
-----
Single subject dataset from the "Neurospin Localizer".
It is a fast event related design:
during 5 minutes, 80 events of the following types are presented :

- 'audio_computation',
- 'audio_left_hand_button_press',
- 'audio_right_hand_button_press',
- 'horizontal_checkerboard',
- 'sentence_listening',
- 'sentence_reading',
- 'vertical_checkerboard',
- 'visual_computation',
- 'visual_left_hand_button_press',
- 'visual_right_hand_button_press'

The data was preprocessed and is in MNI305 space.

The protocol described is the so-called “ARCHI Standard” functional localizer task.

For details on the task, please see :footcite:t:`Pinel2007`.

Direct download link from OSF: https://osf.io/2bqxn

Content
-------
.. nilearn_dataset_content:: localizer_first_level

References
----------
.. footbibliography::

License
-------
.. nilearn_dataset_license:: localizer_first_level
