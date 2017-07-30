# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import matplotlib.pyplot as plt

from nilearn.plotting.matrix_plotting import plot_matrix

##############################################################################
# Some smoke testing for graphics-related code


def test_matrix_plotting():
    from numpy import zeros
    mat = zeros((10, 10))
    labels = str(range(10))
    ax = plot_matrix(mat, labels=labels, title='foo')
    plt.close()
    # test if plotting lower triangle works
    ax = plot_matrix(mat, labels=labels, tri='lower')
    # test if it returns an AxesImage
    ax.axes.set_title('Title')
    plt.close()


