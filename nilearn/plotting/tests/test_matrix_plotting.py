import os

import pytest

import numpy as np
import pandas as pd
from nibabel.tmpdirs import InTemporaryDirectory
import matplotlib.pyplot as plt

from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix)

from nilearn.plotting.matrix_plotting import (
    plot_matrix, plot_contrast_matrix, plot_event, plot_design_matrix)


##############################################################################
# Some smoke testing for graphics-related code


def test_matrix_plotting():
    from numpy import zeros, array
    from distutils.version import LooseVersion
    mat = zeros((10, 10))
    labels = [str(i) for i in range(10)]
    ax = plot_matrix(mat, labels=labels, title='foo')
    plt.close()
    # test if plotting lower triangle works
    ax = plot_matrix(mat, labels=labels, tri='lower')
    # test if it returns an AxesImage
    ax.axes.set_title('Title')
    plt.close()
    ax = plot_matrix(mat, labels=labels, tri='diag')
    ax.axes.set_title('Title')
    plt.close()
    # test if an empty list works as an argument for labels
    ax = plot_matrix(mat, labels=[])
    plt.close()
    # test if an array gets correctly cast to a list
    ax = plot_matrix(mat, labels=array(labels))
    plt.close()
    # test if labels can be None
    ax = plot_matrix(mat, labels=None)
    plt.close()
    pytest.raises(ValueError, plot_matrix, mat, labels=[0, 1, 2])

    import scipy
    if LooseVersion(scipy.__version__) >= LooseVersion('1.0.0'):
        # test if a ValueError is raised when reorder=True without labels
        pytest.raises(ValueError, plot_matrix, mat, labels=None, reorder=True)
        # test if a ValueError is raised when reorder argument is wrong
        pytest.raises(ValueError, plot_matrix, mat, labels=labels, reorder=' ')
        # test if reordering with default linkage works
        idx = [2, 3, 5]
        from itertools import permutations
        # make symmetric matrix of similarities so we can get a block
        for perm in permutations(idx, 2):
            mat[perm] = 1
        ax = plot_matrix(mat, labels=labels, reorder=True)
        assert len(labels) == len(ax.axes.get_xticklabels())
        reordered_labels = [int(lbl.get_text())
                            for lbl in ax.axes.get_xticklabels()]
        # block order does not matter
        assert reordered_labels[:3] == idx or reordered_labels[-3:] == idx, 'Clustering does not find block structure.'
        plt.close()
        # test if reordering with specific linkage works
        ax = plot_matrix(mat, labels=labels, reorder='complete')
        plt.close()


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
