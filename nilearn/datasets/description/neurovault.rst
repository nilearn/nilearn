Neurovault statistical maps


Notes
-----
Neurovault is a public repository of unthresholded statistical
maps, parcellations, and atlases of the human brain. You can read
about it and browse the images it contains at www.neurovault.org.

It is also possible to ask Neurosynth to annotate the maps found on
Neurovault. Neurosynth is a platform for large-scale, automated
synthesis of fMRI data. It can be used to perform decoding.  You can
find out more about Neurosynth at www.neurosynth.org.

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
.. [1] Gorgolewski KJ, Varoquaux G, Rivera G, Schwartz Y, Ghosh SS,
   Maumet C, Sochat VV, Nichols TE, Poldrack RA, Poline J-B, Yarkoni
   T and Margulies DS (2015) NeuroVault.org: a web-based repository
   for collecting and sharing unthresholded statistical maps of the
   human brain. Front. Neuroinform. 9:8.  doi:
   10.3389/fninf.2015.00008

.. [2] Yarkoni, Tal, Russell A. Poldrack, Thomas E. Nichols, David
   C. Van Essen, and Tor D. Wager. "Large-scale automated synthesis
   of human functional neuroimaging data." Nature methods 8, no. 8
   (2011): 665-670.


License
-------
All data are distributed under the CC0 license.
