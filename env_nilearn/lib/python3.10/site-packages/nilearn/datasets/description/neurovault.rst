.. _neurovault_dataset:

Neurovault statistical maps
===========================

Notes
-----
Neurovault is a public repository of unthresholded statistical
maps, parcellations, and atlases of the human brain. You can read
about it and browse the images it contains at www.neurovault.org.

It is also possible to ask Neurosynth to annotate the maps found on
Neurovault. Neurosynth is a platform for large-scale, automated
synthesis of fMRI data. It can be used to perform decoding.  You can
find out more about Neurosynth at www.neurosynth.org.

For more information, see :footcite:t:`Gorgolewski2015`,
and :footcite:t:`Yarkoni2011`.

Content
-------
    :'images': Nifti images representing the statistical maps.
    :'images_meta': Dictionaries containing metadata for each image.
    :'collections_meta': Dictionaries containing metadata for collections.
    :'vocabulary': A list of words retrieved from neurosynth.org
    :'word_frequencies': For each image, the weights of the words
                         from 'vocabulary'.

References
----------

.. footbibliography::

License
-------
All data are distributed under the CC0 license.
