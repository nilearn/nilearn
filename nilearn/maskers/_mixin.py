"""Mixin classes for maskers."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.docs import fill_doc
from nilearn._utils.niimg_conversions import iter_check_niimg
from nilearn._utils.numpy_conversions import csv_to_array
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.image import high_variance_confounds
from nilearn.surface.surface import SurfaceImage
from nilearn.typing import NiimgLike


class _MultiMixin:
    """Mixin class to add common MultiMasker functionalities."""

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(masker=True, multi_masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(masker=True, multi_masker=True)
        return tags

    @fill_doc
    def fit_transform(self, imgs, y=None, confounds=None, sample_mask=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        imgs : Image object, or a :obj:`list` of Image objects
            See :ref:`extracting_data`.
            Data to be preprocessed

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        %(confounds_multi)s

        %(sample_mask_multi)s

            .. versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_multi_nifti)s
        """
        return self.fit(imgs, y=y).transform(
            imgs, confounds=confounds, sample_mask=sample_mask
        )

    @fill_doc
    def transform_imgs(
        self, imgs_list, confounds=None, n_jobs=1, sample_mask=None
    ):
        """Extract signals from a list of 4D niimgs.

        Parameters
        ----------
        %(imgs)s
            Images to process.

        %(confounds_multi)s

        %(n_jobs)s

        %(sample_mask_multi)s

        Returns
        -------
        %(signals_transform_imgs_multi_nifti)s

        """
        # We handle the resampling of maps and mask separately because the
        # affine of the maps and mask images should not impact the extraction
        # of the signal.

        check_is_fitted(self)

        niimg_iter = iter_check_niimg(
            imgs_list,
            ensure_ndim=None,
            atleast_4d=False,
            memory=self.memory_,
            memory_level=self.memory_level,
        )

        confounds = self._prepare_confounds(imgs_list, confounds)

        sample_mask = self._prepare_sample_mask(imgs_list, sample_mask)

        # rely on the transform_single_imgs method
        # defined in each child class
        func = self._cache(self.transform_single_imgs)

        region_signals = Parallel(n_jobs=n_jobs)(
            delayed(func)(imgs=imgs, confounds=cfs, sample_mask=sms)
            for imgs, cfs, sms in zip(
                niimg_iter, confounds, sample_mask, strict=False
            )
        )
        return region_signals

    @fill_doc
    def transform(self, imgs, confounds=None, sample_mask=None):
        """Apply mask, spatial and temporal preprocessing.

        Parameters
        ----------
        imgs :Image object, or a :obj:`list` of Image objects
            See :ref:`extracting_data`.
            Data to be preprocessed

        %(confounds_multi)s

        %(sample_mask_multi)s

        Returns
        -------
        %(signals_transform_multi_nifti)s

        """
        check_is_fitted(self)

        if not (confounds is None or isinstance(confounds, list)):
            raise TypeError(
                "'confounds' must be a None or a list. "
                f"Got {confounds.__class__.__name__}."
            )
        if not (sample_mask is None or isinstance(sample_mask, list)):
            raise TypeError(
                "'sample_mask' must be a None or a list. "
                f"Got {sample_mask.__class__.__name__}."
            )
        if isinstance(imgs, (*NiimgLike, SurfaceImage)):
            if isinstance(confounds, list):
                confounds = confounds[0]
            if isinstance(sample_mask, list):
                sample_mask = sample_mask[0]
            return super().transform(
                imgs, confounds=confounds, sample_mask=sample_mask
            )

        # TODO throw a proper error
        # check we have consistent type
        if isinstance(imgs[0], SurfaceImage):
            assert all(isinstance(x, SurfaceImage) for x in imgs)

        return self.transform_imgs(
            imgs,
            confounds=confounds,
            sample_mask=sample_mask,
            n_jobs=self.n_jobs,
        )

    def _prepare_confounds(self, imgs_list, confounds):
        """Check and prepare confounds."""
        if confounds is None:
            confounds = list(itertools.repeat(None, len(imgs_list)))
        elif len(confounds) != len(imgs_list):
            raise ValueError(
                f"number of confounds ({len(confounds)}) unequal to "
                f"number of images ({len(imgs_list)})."
            )

        if self.high_variance_confounds:
            for i, img in enumerate(imgs_list):
                hv_confounds = self._cache(high_variance_confounds)(img)

                if confounds[i] is None:
                    confounds[i] = hv_confounds
                elif isinstance(confounds[i], list):
                    confounds[i] += hv_confounds
                elif isinstance(confounds[i], np.ndarray):
                    confounds[i] = np.hstack([confounds[i], hv_confounds])
                elif isinstance(confounds[i], pd.DataFrame):
                    confounds[i] = np.hstack(
                        [confounds[i].to_numpy(), hv_confounds]
                    )
                elif isinstance(confounds[i], (str, Path)):
                    c = csv_to_array(confounds[i])
                    if np.isnan(c.flat[0]):
                        # There may be a header
                        c = csv_to_array(confounds[i], skip_header=1)
                    confounds[i] = np.hstack([c, hv_confounds])
                else:
                    confounds[i].append(hv_confounds)

        return confounds

    def _prepare_sample_mask(self, imgs_list, sample_mask):
        """Check and prepare sample_mask."""
        if sample_mask is None:
            sample_mask = itertools.repeat(None, len(imgs_list))
        elif len(sample_mask) != len(imgs_list):
            raise ValueError(
                f"number of sample_mask ({len(sample_mask)}) unequal to "
                f"number of images ({len(imgs_list)})."
            )

        return sample_mask

    def set_output(self, *, transform=None):
        """Set the output container when ``"transform"`` is called.

        .. warning::

            This has not been implemented yet.
        """
        raise NotImplementedError()


class _LabelMaskerMixin:
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features :default=None
            Only for sklearn API compatibility.
        """
        del input_features
        return np.asarray(self.region_names_.values(), dtype=object)
