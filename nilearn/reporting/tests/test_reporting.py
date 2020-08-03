import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nibabel.tmpdirs import InTemporaryDirectory
# Set backend to avoid DISPLAY problems
from nilearn.plotting import _set_mpl_backend

from nilearn.glm.first_level.design_matrix import make_first_level_design_matrix
from nilearn.reporting import (get_clusters_table,
                               plot_contrast_matrix,
                               plot_event,
                               plot_design_matrix,
                               )
from nilearn.reporting._get_clusters_table import _local_max

# Avoid making pyflakes unhappy
_set_mpl_backend
try:
    import matplotlib.pyplot
    # Avoid making pyflakes unhappy
    matplotlib.pyplot
except ImportError:
    have_mpl = False
else:
    have_mpl = True


@pytest.mark.skipif(not have_mpl,
                    reason='Matplotlib not installed; required for this test')
def test_show_design_matrix():
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1., 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model='polynomial', drift_order=3)
    ax = plot_design_matrix(dmtx)
    assert (ax is not None)
    with InTemporaryDirectory():
        ax = plot_design_matrix(dmtx, output_file='dmtx.png')
        assert os.path.exists('dmtx.png')
        assert (ax is None)
        plot_design_matrix(dmtx, output_file='dmtx.pdf')
        assert os.path.exists('dmtx.pdf')


@pytest.mark.skipif(not have_mpl,
                    reason='Matplotlib not installed; required for this test')
def test_show_event_plot():
    # test that the show code indeed (formally) runs
    onset = np.linspace(0, 19., 20)
    duration = np.full(20, 0.5)
    trial_idx = np.arange(20)
    # This makes 11 events in order to test cmap error
    trial_idx[11:] -= 10
    condition_ids = ['a', 'b', 'c', 'd', 'e', 'f',
                     'g', 'h', 'i', 'j', 'k']

    trial_type = np.array([condition_ids[i] for i in trial_idx])

    model_event = pd.DataFrame({
        'onset': onset,
        'duration': duration,
        'trial_type': trial_type
    })
    # Test Dataframe
    fig = plot_event(model_event)
    assert (fig is not None)

    # Test List
    fig = plot_event([model_event, model_event])
    assert (fig is not None)
    
    # Test error
    with pytest.raises(ValueError):
        fig = plot_event(model_event, cmap='tab10')

    # Test save
    with InTemporaryDirectory():
        fig = plot_event(model_event, output_file='event.png')
        assert os.path.exists('event.png')
        assert (fig is None)
        plot_event(model_event, output_file='event.pdf')
        assert os.path.exists('event.pdf')
    

@pytest.mark.skipif(not have_mpl,
                    reason='Matplotlib not installed; required for this test')
def test_show_contrast_matrix():
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1., 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model='polynomial', drift_order=3)
    contrast = np.ones(4)
    ax = plot_contrast_matrix(contrast, dmtx)
    assert (ax is not None)
    with InTemporaryDirectory():
        ax = plot_contrast_matrix(contrast, dmtx, output_file='contrast.png')
        assert os.path.exists('contrast.png')
        assert (ax is None)
        plot_contrast_matrix(contrast, dmtx, output_file='contrast.pdf')
        assert os.path.exists('contrast.pdf')


def test_local_max():
    shape = (9, 10, 11)
    data = np.zeros(shape)
    # Two maxima (one global, one local), 10 voxels apart.
    data[4, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    data[5, 5, :] = [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 6]
    data[6, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    affine = np.eye(4)

    ijk, vals = _local_max(data, affine, min_distance=9)
    assert np.array_equal(ijk, np.array([[5., 5., 10.], [5., 5., 0.]]))
    assert np.array_equal(vals, np.array([6, 5]))

    ijk, vals = _local_max(data, affine, min_distance=11)
    assert np.array_equal(ijk, np.array([[5., 5., 10.]]))
    assert np.array_equal(vals, np.array([6]))


def test_get_clusters_table():
    shape = (9, 10, 11)
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.
    stat_img = nib.Nifti1Image(data, np.eye(4))

    # test one cluster extracted
    cluster_table = get_clusters_table(stat_img, 4, 0)
    assert len(cluster_table) == 1

    # test empty table on high stat threshold
    cluster_table = get_clusters_table(stat_img, 6, 0)
    assert len(cluster_table) == 0

    # test empty table on high cluster threshold
    cluster_table = get_clusters_table(stat_img, 4, 9)
    assert len(cluster_table) == 0

def test_get_clusters_table_not_modifying_stat_image():
    shape = (9, 10, 11)
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.
    data[0:3, 0:3, 0:3] = 6.

    stat_img = nib.Nifti1Image(data, np.eye(4))
    data_orig = image.get_data(stat_img).copy()

    # test one cluster should be removed 
    clusters_table = get_clusters_table(stat_img, 4, cluster_threshold=10)
    assert np.allclose(data_orig, image.get_data(stat_img))
    assert len(clusters_table) == 1

    # test no clusters should be removed
    stat_img = nib.Nifti1Image(data, np.eye(4))
    clusters_table = get_clusters_table(stat_img, 4, cluster_threshold=7)
    assert np.allclose(data_orig, image.get_data(stat_img))
    assert len(clusters_table) == 2

    # test cluster threshold is None
    stat_img = nib.Nifti1Image(data, np.eye(4))
    clusters_table = get_clusters_table(stat_img, 4, cluster_threshold=None)
    assert np.allclose(data_orig, image.get_data(stat_img))
    assert len(clusters_table) == 2