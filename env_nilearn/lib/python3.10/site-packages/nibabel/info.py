"""Define static nibabel metadata for nibabel

The long description parameter is used in the nibabel top-level docstring,
and in building the docs.
We exec this file in several places, so it cannot import nibabel or use
relative imports.
"""

# Note: this long_description is the canonical place to edit this text.
# It also appears in README.rst, but it should get there by running
# ``tools/refresh_readme.py`` which pulls in this version.
# We also include this text in the docs by ``..include::`` in
# ``docs/source/index.rst``.
long_description = """
Read and write access to common neuroimaging file formats, including:
ANALYZE_ (plain, SPM99, SPM2 and later), GIFTI_, NIfTI1_, NIfTI2_, `CIFTI-2`_,
MINC1_, MINC2_, `AFNI BRIK/HEAD`_, ECAT_ and Philips PAR/REC.
In addition, NiBabel also supports FreeSurfer_'s MGH_, geometry, annotation and
morphometry files, and provides some limited support for DICOM_.

NiBabel's API gives full or selective access to header information (metadata),
and image data is made available via NumPy arrays. For more information, see
NiBabel's `documentation site`_ and `API reference`_.

.. _API reference: https://nipy.org/nibabel/api.html
.. _AFNI BRIK/HEAD: https://afni.nimh.nih.gov/pub/dist/src/README.attributes
.. _ANALYZE: http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
.. _CIFTI-2: https://www.nitrc.org/projects/cifti/
.. _DICOM: http://medical.nema.org/
.. _documentation site: http://nipy.org/nibabel
.. _ECAT: http://xmedcon.sourceforge.net/Docs/Ecat
.. _Freesurfer: https://surfer.nmr.mgh.harvard.edu
.. _GIFTI: https://www.nitrc.org/projects/gifti
.. _MGH: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
.. _MINC1:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC1_File_Format_Reference
.. _MINC2:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference
.. _NIfTI1: http://nifti.nimh.nih.gov/nifti-1/
.. _NIfTI2: http://nifti.nimh.nih.gov/nifti-2/

Installation
============

To install NiBabel's `current release`_ with ``pip``, run::

   pip install nibabel

To install the latest development version, run::

   pip install git+https://github.com/nipy/nibabel

When working on NiBabel itself, it may be useful to install in "editable" mode::

   git clone https://github.com/nipy/nibabel.git
   pip install -e ./nibabel

For more information on previous releases, see the `release archive`_ or
`development changelog`_.

.. _current release: https://pypi.python.org/pypi/NiBabel
.. _release archive: https://github.com/nipy/NiBabel/releases
.. _development changelog: https://nipy.org/nibabel/changelog.html

Testing
=======

During development, we recommend using tox_ to run nibabel tests::

    git clone https://github.com/nipy/nibabel.git
    cd nibabel
    tox

To test an installed version of nibabel, install the test dependencies
and run pytest_::

    pip install nibabel[test]
    pytest --pyargs nibabel

For more information, consult the `developer guidelines`_.

.. _tox: https://tox.wiki
.. _pytest: https://docs.pytest.org
.. _developer guidelines: https://nipy.org/nibabel/devel/devguide.html

Mailing List
============

Please send any questions or suggestions to the `neuroimaging mailing list
<https://mail.python.org/mailman/listinfo/neuroimaging>`_.

License
=======

NiBabel is licensed under the terms of the `MIT license
<https://github.com/nipy/nibabel/blob/master/COPYING#nibabel>`__.
Some code included with NiBabel is licensed under the `BSD license`_.
For more information, please see the COPYING_ file.

.. _BSD license: https://opensource.org/licenses/BSD-3-Clause
.. _COPYING: https://github.com/nipy/nibabel/blob/master/COPYING

Citation
========

NiBabel releases have a Zenodo_ `Digital Object Identifier`_ (DOI) badge at
the top of the release notes. Click on the badge for more information.

.. _Digital Object Identifier: https://en.wikipedia.org/wiki/Digital_object_identifier
.. _zenodo: https://zenodo.org
"""
