.. _localizer_first_level_dataset:

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

Content
-------
The dataset includes
    :'epi_img': the input 4D image
    :'events': a csv file describing the paradigm
    :'description': data description

References
----------

License
-------
unknown
