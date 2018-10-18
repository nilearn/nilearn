# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import tempfile

import matplotlib.pyplot as plt
import nibabel
import numpy as np

from nilearn.plotting.displays import OrthoSlicer, XSlicer, OrthoProjector
from nilearn.plotting.displays import LZRYProjector
from nilearn.datasets import load_mni152_template

##############################################################################
# Some smoke testing for graphics-related code


def test_demo_ortho_slicer():
    # This is only a smoke test
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    img = load_mni152_template()
    oslicer.add_overlay(img, cmap=plt.cm.gray)
    oslicer.close()


def test_stacked_slicer():
    # Test stacked slicers, like the XSlicer
    img = load_mni152_template()
    slicer = XSlicer.init_with_figure(img=img, cut_coords=3)
    slicer.add_overlay(img, cmap=plt.cm.gray)
    # Forcing a layout here, to test the locator code
    with tempfile.TemporaryFile() as fp:
        slicer.savefig(fp)
    slicer.close()


def test_demo_ortho_projector():
    # This is only a smoke test
    img = load_mni152_template()
    oprojector = OrthoProjector.init_with_figure(img=img)
    oprojector.add_overlay(img, cmap=plt.cm.gray)
    with tempfile.TemporaryFile() as fp:
        oprojector.savefig(fp)
    oprojector.close()


def test_contour_fillings_levels_in_add_contours():
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    img = load_mni152_template()
    # levels should be atleast 2
    # If single levels are passed then we force upper level to be inf
    oslicer.add_contours(img, filled=True, colors='r',
                         alpha=0.2, levels=[0.])

    # If two levels are passed, it should be increasing from zero index
    # In this case, we simply omit appending inf
    oslicer.add_contours(img, filled=True, colors='b',
                         alpha=0.1, levels=[0., 0.2])

    # without passing colors and alpha. In this case, default values are
    # chosen from matplotlib
    oslicer.add_contours(img, filled=True, levels=[0., 0.2])

    # levels with only one value
    oslicer.add_contours(img, filled=True, levels=[0.])

    # without passing levels, should work with default levels from
    # matplotlib
    oslicer.add_contours(img, filled=True)


def test_user_given_cmap_with_colorbar():
    img = load_mni152_template()
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))

    # Test with cmap given as a string
    oslicer.add_overlay(img, cmap='Paired', colorbar=True)
    oslicer.close()


def test_data_complete_mask():
    """This special case test is due to matplotlib 2.1.0.

    When the data is completely masked, then we have plotting issues
    See similar issue #9280 reported in matplotlib. This function
    tests the patch added for this particular issue.
    """
    # data is completely masked
    data = np.zeros((10, 20, 30))
    affine = np.eye(4)

    img = nibabel.Nifti1Image(data, affine)
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    oslicer.add_overlay(img)
    oslicer.close()


def test_add_markers_cut_coords_is_none():
    # A special case test for add_markers when cut_coords are None. This
    # case is used when coords are placed on glass brain
    orthoslicer = OrthoSlicer(cut_coords=(None, None, None))
    orthoslicer.add_markers([(0, 0, 2)])
    orthoslicer.close()


def test_annotations():
    # Check calls to display.annotate()
    # In particular, exercise some of the keyword arguments for scale bars
    orthoslicer = OrthoSlicer(cut_coords=(None, None, None))
    orthoslicer.annotate(size=10, left_right=True, positions=False)
    orthoslicer.annotate(size=12, left_right=False, positions=False,
                         scalebar=True,
                         scale_size=2.5,
                         scale_units='cm',
                         scale_loc=3)
    orthoslicer.close()


def test_add_graph_with_node_color_as_string():
    lzry_projector = LZRYProjector(cut_coords=(0, 0, 0, 0))
    matrix = np.array([[0, 3], [3, 0]])
    node_coords = [[-53.60, -62.80, 36.64], [23.87, 0.31, 69.42]]
    # node_color as string
    lzry_projector.add_graph(matrix, node_coords, node_color='red')
    lzry_projector.close()
    # node_color as sequence of string
    lzry_projector.add_graph(matrix, node_coords, node_color=['red', 'blue'])
    lzry_projector.close()
