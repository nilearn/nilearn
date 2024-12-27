.. _neurovault:

===========================================================
Downloading statistical maps from the Neurovault repository
===========================================================

Neurovault is a public repository of unthresholded statistical maps,
parcellations, and atlases of the human brain. You can read about it
and browse the images it contains at https://www.neurovault.org. You
can download maps from Neurovault with Nilearn.

Neurovault was introduced in :footcite:t:`Gorgolewski2015`.

Neurovault contains collections of images. We can get information
about each collection - such as who uploaded it, a link to a paper, a
description - and about each image - the modality, number of subjects,
some tags, and more. The nilearn downloaders will fetch this metadata
and the images themselves.

Nilearn provides two functions to download statistical maps from
Neurovault.

Specific images or collections
------------------------------

In the simplest case, you already know the "id" of the collections or
images you want. Maybe you liked a paper and went to
https://www.neurovault.org looking for the data. Once on the relevant
collection's webpage, you can click 'Details' to see its id
(and more). You can then download it using
:func:`nilearn.datasets.fetch_neurovault_ids` :

.. code-block:: python

  from nilearn.datasets import fetch_neurovault_ids
  brainpedia = fetch_neurovault_ids(collection_ids=[1952])

Or if you want some images in particular, rather than whole
collections :

.. code-block:: python

  brainpedia_subset = fetch_neurovault_ids(image_ids=[32015, 32016])

Selection filters
-----------------

You may not know which collections or images you want. For example,
you may be conducting a meta-analysis and want to grab all the images
that are related to "language". Using
:func:`nilearn.datasets.fetch_neurovault`, you can fetch all the images and
collections that match your criteria - you don't need to know their
ids.

The filters are applied to images' and collections' metadata.

You can describe filters with dictionaries. Each collection's
metadata is compared to the parameter ``collection_terms``. Collections
for which ``collection_metadata['key'] == value`` is not ``True`` for
every key, value pair in ``collection_terms`` will be discarded. We use
``image_terms`` in the same way to filter images.

For example, many images on Neurovault have a "modality" field in their
metadata.  :term:`BOLD` images should have it set to "fMRI-BOLD".
We can ask for :term:`BOLD` images only :

.. code-block:: python

  bold = fetch_neurovault(image_terms={'modality': 'fMRI-BOLD'}, max_images=7)

Here we set the max_images parameter to 7, so that you can try this snippet
without waiting for a long time. To get all the images which match your
filters, you should set max_images to ``None``, which means "get as many
images as possible". The default for max_images is 100.

The default values for the ``collection_terms`` and ``image_terms`` parameters
filter out empty collections, and exclude an image if one of the following is
true:

- it is not in :term:`MNI` space.
- its metadata field "is_valid" is cleared.
- it is thresholded.
- its map type is one of "ROI/mask", "anatomical", or "parcellation".
- its image type is "atlas"

Extra keyword arguments are treated as additional image filters, so if we want
to keep the default filters, and add the requirement that the modality should
be "fMRI-BOLD", we can write:

.. code-block:: python

  bold = fetch_neurovault(modality='fMRI-BOLD', max_images=7)

Sometimes the selection criteria are more complex than a simple
comparison to a single value. For example, we may also be interested
in CBF and CBV images. In ``nilearn``, the ``dataset.neurovault`` module
provides ``IsIn`` which makes this easy :

.. code-block:: python

  from nilearn.datasets import neurovault
  fmri = fetch_neurovault(modality=neurovault.IsIn('fMRI-BOLD', 'fMRI-CBF', 'fMRI-CBV'),
                          max_images=100)

We could also have used ``Contains`` :

.. code-block:: python

  fmri = fetch_neurovault(modality=neurovault.Contains('fMRI'), max_images=7)

If we need regular expressions, we can also use ``Pattern`` :

.. code-block:: python

  fmri = fetch_neurovault(
    modality=neurovault.Pattern('fmri(-.*)?', neurovault.re.IGNORECASE),
    max_images=7)

The complete list of such special values available in
``nilearn.datasets.neurovault`` is:
``IsNull``, ``NotNull``, ``NotEqual``, ``GreaterOrEqual``,
``GreaterThan``, ``LessOrEqual``, ``LessThan``, ``IsIn``, ``NotIn``,
``Contains``, ``NotContains``, ``Pattern``.

You can also use ``ResultFilter`` to easily express boolean logic
(AND, OR, XOR, NOT).


**If you need more complex filters**, and using dictionaries as shown above is
not convenient, you can express filters as functions. The parameter
``collection_filter`` should be a callable, which will be called once for each
collection. The sole argument will be a dictionary containing the metadata for
the collection. The filter should return ``True`` if the collection is to be
kept, and ``False`` if it is to be discarded. ``image_filter`` does the same
job for images. The default values for these parameters don't filter out
anything.
Using a filter rather than a dictionary, the first example becomes:

.. code-block:: python

  bold = fetch_neurovault(image_filter=lambda meta: meta.get('modality') == 'fMRI-BOLD',
                          image_terms={}, max_images=7)

.. note::

  Even if you specify a filter as a function, the default filters for
  ``image_terms`` and ``collection_terms`` still apply; pass an empty
  dictionary if you want to disable them. Without ``image_terms={}`` in the
  call above, parcellations, images not in :term:`MNI` space, etc. would be still be
  filtered out.


The example above can be rewritten using dictionaries, but in some cases you
will need to use ``image_filter`` or ``collection_filter``. For example,
suppose that for some weird reason you only want images that don't have too
many metadata fields - say, an image should only be kept if its metadata has
less than 50 fields.  This cannot be done by simply comparing each key in a
metadata dictionary to a required value, so we need to write our own filter:

.. code-block:: python

  small_meta_images = fetch_neurovault(image_filter=lambda meta: len(meta) < 50,
                                       max_images=7)


Output
------

Both functions return a dict-like object which exposes its items as
attributes.

It contains:

- ``images``, the paths to downloaded files.
- ``images_meta``, the metadata for the images in a list of dictionaries.
- ``collections_meta``, the metadata for the collections.
- ``description``, a short description of the Neurovault dataset.

Note to ``pandas`` users: passing ``images_meta`` or ``collections_meta``
to the ``DataFrame`` constructor yields the expected result, with
images (or collections) as rows and metadata fields as columns.

Neurosynth annotations
----------------------

It is also possible to ask Neurosynth to annotate the maps found on
Neurovault. Neurosynth is a platform for large-scale, automated
synthesis of :term:`fMRI` data. It can be used to perform decoding.  You can
learn more about Neurosynth at https://www.neurosynth.org.

Neurosynth was introduced in :footcite:`Yarkoni2011`.

If you set the parameter ``fetch_neurosynth_words`` when calling
``fetch_neurovault`` or ``fetch_neurovault_ids``, we will also
download the annotations for the resulting images. They will be stored
as json files on your disk. The result will also contain (unless you
clear the ``vectorize_words`` parameter to save computation time):

- ``vocabulary``, a list of words
- ``word_frequencies``, the weight of the words returned by
  neurosynth.org for each image, such that the weight of word
  ``vocabulary[j]`` for the image found in ``images[i]`` is
  ``word_frequencies[i, j]``

Examples using Neurovault
-------------------------

- :ref:`sphx_glr_auto_examples_07_advanced_plot_ica_neurovault.py`
      Download images from Neurovault and extract some networks using ICA.

- :ref:`sphx_glr_auto_examples_07_advanced_plot_neurovault_meta_analysis.py`
    Meta-analysis of "Stop minus go" studies available on Neurovault.

References
----------

.. footbibliography::
