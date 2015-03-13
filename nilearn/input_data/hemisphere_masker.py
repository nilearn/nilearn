plotted_subject = 0  # subject to plot
n_jobs = 1

import numpy as np

import matplotlib.pyplot as plt
import nibabel
from sklearn.externals.joblib import Memory

from nilearn.image import iter_img, reorder_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map


class HemisphereMasker(NiftiMasker):
    """
    Masker to segregate by hemisphere.

    Parameters
    ==========
    hemisphere: L or R

    """
    def __init__(self, mask_img=None, sessions=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background',
                 mask_args=None, sample_mask=None,
                 memory_level=1, memory=Memory(cachedir=None),
                 verbose=0, hemisphere='L'):
        if hemisphere.lower() in ['l', 'left']:
            self.hemi = 'l'
        elif hemisphere.lower() in ['r', 'right']:
            self.hemi = 'r'
        else:
            raise ValueError('Hemisphere must be left or right; '
                             'got value %s' % self.hemi)

        super(HemisphereMasker, self).__init__(mask_img=mask_img,
                                               sessions=sessions,
                                               smoothing_fwhm=smoothing_fwhm,
                                               standardize=standardize,
                                               detrend=detrend,
                                               low_pass=low_pass,
                                               high_pass=high_pass,
                                               t_r=t_r,
                                               target_affine=target_affine,
                                               target_shape=target_shape,
                                               mask_strategy=mask_strategy,
                                               mask_args=mask_args,
                                               sample_mask=sample_mask,
                                               memory_level=memory_level,
                                               memory=memory,
                                               verbose=verbose)

    def fit(self, X=None, y=None):
        super(HemisphereMasker, self).fit(X, y)

        # x, y, z
        hemi_mask_data = reorder_img(self.mask_img_).get_data().astype(np.bool)

        xvals = hemi_mask_data.shape[0]
        midpt = np.ceil(xvals / 2.)
        if self.hemi == 'l':
            other_hemi_slice = slice(midpt, xvals)
        else:
            other_hemi_slice = slice(0, midpt)

        hemi_mask_data[other_hemi_slice] = False
        mask_data = self.mask_img_.get_data() * hemi_mask_data
        self.mask_img_ = nibabel.Nifti1Image(mask_data,
                                             self.mask_img_.get_affine())

        return self


def split_bilateral_rois(maps_img, show_results=False):
    """Convenience function for splitting bilateral ROIs
    into two unilateral ROIs"""

    new_rois = []

    for map_img in iter_img(maps_img):
        if show_results:
            plot_stat_map(map_img, title='raw')
        for hemi in ['L', 'R']:
            hemi_mask = HemisphereMasker(hemisphere=hemi)
            hemi_mask.fit(map_img)
            if hemi_mask.mask_img_.get_data().sum() > 0:
                hemi_vectors = hemi_mask.transform(map_img)
                hemi_img = hemi_mask.inverse_transform(hemi_vectors)
                new_rois.append(hemi_img.get_data())
                if show_results:
                    plot_stat_map(hemi_img, title=hemi)
        if show_results:
            plt.show()
    new_maps_data = np.concatenate(new_rois, axis=3)
    new_maps_img = nibabel.Nifti1Image(new_maps_data, maps_img.get_affine())
    print ("Changed from %d ROIs to %d ROIs" % (maps_img.shape[-1],
                                                new_maps_img.shape[-1]))
    return new_maps_img
