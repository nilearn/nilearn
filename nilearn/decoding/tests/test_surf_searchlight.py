"""
Test the searchlight module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import numpy as np
import nibabel
import sklearn
from distutils.version import LooseVersion
from nose.tools import assert_equal

import nibabel as nb
from nibabel import gifti


from ...surface.tests.test_surface import _generate_surf
from nilearn.decoding.surf_searchlight import SurfSearchLight


def test_surf_searchlight():
    # Create a toy dataset to run searchlight on

    # Generate a mesh
    mesh = _generate_surf()
    n_vertex = mesh[0].shape[0]
    coord_array = gifti.GiftiDataArray(data=mesh[0])
    coord_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET']

    face_array = gifti.GiftiDataArray(data=mesh[1])
    face_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']

    gii_mesh = gifti.GiftiImage(darrays=[coord_array, face_array])

    # Generate full-mesh mask
    data = np.ones(n_vertex)
    if LooseVersion(nb.__version__) > LooseVersion('2.0.2'):
        darray = gifti.GiftiDataArray(data=data)
    else:
        # Avoid a bug in nibabel 1.2.0 where GiftiDataArray were not
        # initialized properly:
        darray = gifti.GiftiDataArray.from_array(data,
                                                 intent='t test')
    gii_mask_tex = gifti.GiftiImage(darrays=[darray])

    # Create searchlight object
    searchlight = SurfSearchLight(gii_mesh,
                                  gii_mask_tex,
                                  process_surfmask_tex=gii_mask_tex,
                                  cv=4)

    # Generate functional data on this mesh
    rng = np.random.RandomState(0)
    frames = 20
    X = rng.rand(n_vertex, frames)

    # Generate labels for decoding
    y = np.arange(frames, dtype=int) > (frames // 2)

    # Run searchlight decoding
    scores = searchlight.fit(X,y)