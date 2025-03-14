.. _language_localizer_dataset:

language localizer demo dataset
===============================

Access
------
See :func:`nilearn.datasets.fetch_language_localizer_demo_dataset`.

Notes
-----
10 subjects were scanned with fMRI during a "language localizer"
where they (covertly) read meaningful sentences (trial_type='language')
or strings of consonants (trial_type='string'),
presented one word at a time at the center of the screen (rapid serial visual presentation).

The functional images files (in derivatives/)
have been preprocessed (spatially realigned and normalized into the :term:`MNI` space).
Initially acquired with a :term:`voxel` size of 1.5x1.5x1.5mm,
they have been resampled to 4.5x4.5x4.5mm to save disk space.

https://osf.io/k4jp8/


# This task, described in Pinel et al., BMC neuroscience 2007 probes basic
# functions, such as button presses with the left or right hand, viewing
# horizontal and vertical checkerboards, reading and listening to short
# sentences, and mental computations (subtractions).
#
# Visual stimuli were displayed in four 250-ms epochs, separated by 100ms
# intervals (i.e., 1.3s in total). Auditory stimuli were drawn from a recorded
# male voice (i.e., a total of 1.6s for motor instructions, 1.2-1.7s for
# sentences, and 1.2-1.3s for subtractions). The auditory or visual stimuli
# were shown to the participants for passive listening or viewing or responses
# via button presses in event-related paradigms.  Post-scan questions verified
# that the experimental tasks were understood and followed correctly.
#
# This task comprises 10 conditions:
#
# * audio_left_hand_button_press: Left-hand three-times button press,
#   indicated by auditory instruction
# * audio_right_hand_button_press: Right-hand three-times button press,
#   indicated by auditory instruction
# * visual_left_hand_button_press: Left-hand three-times button press,
#   indicated by visual instruction
# * visual_right_hand_button_press:  Right-hand three-times button press,
#   indicated by visual instruction
# * horizontal_checkerboard: Visualization of flashing horizontal checkerboards
# * vertical_checkerboard: Visualization of flashing vertical checkerboards
# * sentence_listening: Listen to narrative sentences
# * sentence_reading: Read narrative sentences
# * audio_computation: Mental subtraction, indicated by auditory instruction
# * visual_computation: Mental subtraction, indicated by visual instruction

Content
-------
    :'data_dir': Path to downloaded dataset.

    :'downloaded_files': Absolute paths of downloaded files on disk

        - epi_img: the input 4D image

        - events: a csv file describing the paradigm

        - description: data description

        - t_r: repetition time in seconds

References
----------


License
-------
ODC-BY-SA
