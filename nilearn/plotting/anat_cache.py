# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
3D visualization of activation maps using Mayavi

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

# Standard library imports
import os

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
from scipy import ndimage

from nibabel import load

# The sform for MNI templates
mni_sform = np.array([[-1, 0, 0,   90],
                      [ 0, 1, 0, -126],
                      [ 0, 0, 1,  -72],
                      [ 0, 0, 0,   1]])

mni_sform_inv = np.linalg.inv(mni_sform)

def find_mni_template():
    """ Try to find an MNI template on the disk.
    """
    from nipy.utils import templates, DataError
    try:
        filename = templates.get_filename(
                            'ICBM152', '1mm', 'T1_brain.nii.gz')
        if os.path.exists(filename):
            return filename
    except DataError:
        pass
    possible_paths = [
     ('', 'usr', 'share', 'fsl', 'data', 'standard', 'avg152T1_brain.nii.gz'),
     ('', 'usr', 'share', 'data', 'fsl-mni152-templates', 'avg152T1_brain.nii.gz'),
     ('', 'usr', 'local', 'share', 'fsl', 'data', 'standard', 'avg152T1_brain.nii.gz'),
            ]
    if 'FSLDIR' in os.environ:
        fsl_path = os.environ['FSLDIR'].split(os.sep)
        fsl_path.extend(('data', 'standard', 'avg152T1_brain.nii.gz'))
        possible_paths.append(fsl_path)
    for path in possible_paths:
        filename = os.sep.join((path))
        if os.path.exists(filename):
            return filename



################################################################################
# Caching of the MNI template.
################################################################################

class _AnatCache(object):
    """ Class to store the anat array in cache, to avoid reloading it
        each time.
    """
    anat        = None
    anat_sform  = None
    blurred     = None

    @classmethod
    def get_anat(cls):
        filename = find_mni_template()
        if cls.anat is None:
            if filename is None:
                raise OSError('Cannot find template file T1_brain.nii.gz '
                        'required to plot anatomy, see the nipy documentation '
                        'installaton section for how to install template files.')
            anat_im = load(filename)
            anat = anat_im.get_data()
            anat = anat.astype(np.float)
            anat_mask = ndimage.morphology.binary_fill_holes(anat > 0)
            anat = np.ma.masked_array(anat, np.logical_not(anat_mask))
            cls.anat_sform = anat_im.get_affine()
            cls.anat_im = anat_im
            cls.anat_max = anat.max()
        return cls.anat_im, cls.anat_max




