.. note::

       If you are using Nilearn with a version older than ``0.9.0``,
       then you should either upgrade your version or import maskers
       from the ``input_data`` module instead of the ``maskers`` module.

       That is, you should manually replace in the following example
       all occurrences of:

       .. code-block:: python

           from nilearn.maskers import NiftiMasker

       with:

       .. code-block:: python

           from nilearn.input_data import NiftiMasker
