"""Generate mask and testing tractogram in known formats:

* mask: standard.nii.gz
* tractogram:

    * standard.trk
"""

import numpy as np

import nibabel as nib
from nibabel.streamlines import FORMATS
from nibabel.streamlines.header import Field


def mark_the_spot(mask):
    """Marks every nonzero voxel using streamlines to form a 3D 'X' inside.

    Generates streamlines forming a 3D 'X' inside every nonzero voxel.

    Parameters
    ----------
    mask : ndarray
        Mask containing the spots to be marked.

    Returns
    -------
    list of ndarrays
        All streamlines needed to mark every nonzero voxel in the `mask`.
    """

    def _gen_straight_streamline(start, end, steps=3):
        coords = []
        for s, e in zip(start, end):
            coords.append(np.linspace(s, e, steps))

        return np.array(coords).T

    # Generate a 3D 'X' template fitting inside the voxel centered at (0,0,0).
    X = [
        _gen_straight_streamline((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
        _gen_straight_streamline((-0.5, 0.5, -0.5), (0.5, -0.5, 0.5)),
        _gen_straight_streamline((-0.5, 0.5, 0.5), (0.5, -0.5, -0.5)),
        _gen_straight_streamline((-0.5, -0.5, 0.5), (0.5, 0.5, -0.5)),
    ]

    # Get the coordinates of voxels 'on' in the mask.
    coords = np.array(zip(*np.where(mask)))

    streamlines = [(line + c) * voxel_size for c in coords for line in X]

    return streamlines


if __name__ == '__main__':
    rng = np.random.RandomState(42)

    width = 4  # Coronal
    height = 5  # Sagittal
    depth = 7  # Axial

    voxel_size = np.array((1.0, 3.0, 2.0))

    # Generate a random mask with voxel order RAS+.
    mask = rng.rand(width, height, depth) > 0.8
    mask = (255 * mask).astype(np.uint8)

    # Build tractogram
    streamlines = mark_the_spot(mask)
    tractogram = nib.streamlines.Tractogram(streamlines)

    # Build header
    affine = np.eye(4)
    affine[range(3), range(3)] = voxel_size
    header = {
        Field.DIMENSIONS: (width, height, depth),
        Field.VOXEL_SIZES: voxel_size,
        Field.VOXEL_TO_RASMM: affine,
        Field.VOXEL_ORDER: 'RAS',
    }

    # Save the standard mask.
    nii = nib.Nifti1Image(mask, affine=affine)
    nib.save(nii, 'standard.nii.gz')

    # Save the standard tractogram in every available file format.
    for ext, cls in FORMATS.items():
        tfile = cls(tractogram, header)
        nib.streamlines.save(tfile, 'standard' + ext)
