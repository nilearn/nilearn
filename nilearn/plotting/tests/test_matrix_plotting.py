# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import matplotlib.pyplot as plt
from nose.tools import assert_true
from nilearn.plotting.matrix_plotting import plot_matrix

##############################################################################
# Some smoke testing for graphics-related code


def test_matrix_plotting():
    from numpy import zeros
    mat = zeros((10, 10))
    labels = [str(i) for i in range(10)]
    ax = plot_matrix(mat, labels=labels, title='foo')
    plt.close()
    # test if plotting lower triangle works
    ax = plot_matrix(mat, labels=labels, tri='lower')
    # test if it returns an AxesImage
    ax.axes.set_title('Title')
    plt.close()
    import scipy
    if int(scipy.__version__.split('.')[0]) >= 1:
        # test if reordering with default linkage works
        idx = [2, 3, 5]
        from itertools import permutations
        # make symmetric matrix of similarities so we can get a block
        for perm in permutations(idx, 2):
            mat[perm] = 1
        ax = plot_matrix(mat, labels=labels, reorder=True)
        reordered_labels = [int(lbl.get_text())
                            for lbl in ax.axes.get_xticklabels()]
        # block order does not matter
        assert_true(reordered_labels[:3] == idx or reordered_labels[-3:] == idx,
                    'Clustering does not find block structure.')
        plt.close()
