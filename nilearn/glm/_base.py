from collections import OrderedDict

import numpy as np
from nibabel.onetime import auto_attr
from sklearn.base import BaseEstimator, TransformerMixin

from nilearn._utils import CacheMixin
from nilearn.maskers import SurfaceMasker
from nilearn.surface import SurfaceImage


class BaseGLM(TransformerMixin, CacheMixin, BaseEstimator):
    """Implement a base class \
    for the :term:`General Linear Model<GLM>`.
    """

    def _is_volume_glm(self):
        """Return if model is run on volume data or not."""
        return not (
            isinstance(self.mask_img, (SurfaceMasker, SurfaceImage))
            or (
                self.__sklearn_is_fitted__()
                and isinstance(self.masker_, SurfaceMasker)
            )
        )

    def _attributes_to_dict(self):
        """Return dict with pertinent model attributes & information.

        Returns
        -------
        dict
        """
        selected_attributes = [
            "subject_label",
            "drift_model",
            "hrf_model",
            "standardize",
            "noise_model",
            "t_r",
            "signal_scaling",
            "scaling_axis",
            "smoothing_fwhm",
            "slice_time_ref",
        ]
        if self._is_volume_glm():
            selected_attributes.extend(["target_shape", "target_affine"])
        if hasattr(self, "hrf_model") and self.hrf_model == "fir":
            selected_attributes.append("fir_delays")
        if hasattr(self, "drift_model"):
            if self.drift_model == "cosine":
                selected_attributes.append("high_pass")
            elif self.drift_model == "polynomial":
                selected_attributes.append("drift_order")

        selected_attributes.sort()

        model_param = OrderedDict(
            (attr_name, getattr(self, attr_name))
            for attr_name in selected_attributes
            if getattr(self, attr_name, None) is not None
        )

        for k, v in model_param.items():
            if isinstance(v, np.ndarray):
                model_param[k] = v.tolist()

        return model_param

    # @auto_attr store the value as an object attribute after initial call
    # better performance than @property
    @auto_attr
    def residuals(self):
        """Transform voxelwise residuals to the same shape \
        as the input Nifti1Image(s).

        Returns
        -------
        output : list
            A list of Nifti1Image(s).

        """
        return self._get_voxelwise_model_attribute(
            "residuals", result_as_time_series=True
        )

    @auto_attr
    def predicted(self):
        """Transform voxelwise predicted values to the same shape \
        as the input Nifti1Image(s).

        Returns
        -------
        output : list
            A list of Nifti1Image(s).

        """
        return self._get_voxelwise_model_attribute(
            "predicted", result_as_time_series=True
        )

    @auto_attr
    def r_square(self):
        """Transform voxelwise r-squared values to the same shape \
        as the input Nifti1Image(s).

        Returns
        -------
        output : list
            A list of Nifti1Image(s).

        """
        return self._get_voxelwise_model_attribute(
            "r_square", result_as_time_series=False
        )

    def generate_report(
        self,
        contrasts=None,
        title=None,
        bg_img="MNI152TEMPLATE",
        threshold=3.09,
        alpha=0.001,
        cluster_threshold=0,
        height_control="fpr",
        two_sided=False,
        min_distance=8.0,
        plot_type="slice",
        cut_coords=None,
        display_mode=None,
        report_dims=(1600, 800),
        input=None,
        verbose=1,
    ):
        """Return a :class:`~nilearn.reporting.HTMLReport` \
        which shows all important aspects of a fitted :term:`GLM`.

        The :class:`~nilearn.reporting.HTMLReport` can be opened in a
        browser, displayed in a notebook, or saved to disk as a standalone
        HTML file.

        The :term:`GLM` must be fitted and have the computed design
        matrix(ces).

        .. note::

            The :class:`~nilearn.glm.first_level.FirstLevelModel` or
            :class:`~nilearn.glm.second_level.SecondLevelModel` must have
            been fitted prior to calling ``generate_report``.

        Parameters
        ----------
        contrasts : :obj:`dict` [ :obj:`str`, :class:`~numpy.ndarray` ] or\
        :obj:`str` or :obj:`list` [ :obj:`str` ] or :class:`~numpy.ndarray` or\
        :obj:`list` [ :class:`~numpy.ndarray` ]

            Contrasts information for a
            :class:`~nilearn.glm.first_level.FirstLevelModel` or
            :class:`~nilearn.glm.second_level.SecondLevelModel`.

            Example:

                Dict of contrast names and coefficients,
                or list of contrast names
                or list of contrast coefficients
                or contrast name
                or contrast coefficient

                Each contrast name must be a string.
                Each contrast coefficient must be a list or
                numpy array of ints.

            Contrasts are passed to ``contrast_def`` for
            :class:`~nilearn.glm.first_level.FirstLevelModel` through
            :meth:`~nilearn.glm.first_level.FirstLevelModel.compute_contrast`,
            and ``second_level_contrast`` for
            :class:`~nilearn.glm.second_level.SecondLevelModel` through
            :meth:`~nilearn.glm.second_level.SecondLevelModel.compute_contrast`.

        title : :obj:`str`, optional
            - If a :obj:`str`, it represents the web page's title and primary
              heading, model type is sub-heading.
            - If ``None``, page titles and headings are autogenerated using
              contrast names.

        bg_img : Niimg-like object, default='MNI152TEMPLATE'
            See :ref:`extracting_data`.
            The background image for mask and stat maps to be plotted on upon.
            To turn off background image, just pass "bg_img=None".

        threshold : :obj:`float`, default=3.09
            Cluster forming threshold in same scale as ``stat_img`` (either a
            t-scale or z-scale value). Used only if ``height_control`` is
            ``None``.

        alpha : :obj:`float`, default=0.001
            Number controlling the thresholding (either a p-value or q-value).
            Its actual meaning depends on the ``height_control`` parameter.
            This function translates alpha to a z-scale threshold.

        cluster_threshold : :obj:`int`, default=0
            Cluster size threshold, in :term:`voxels<voxel>`.

        height_control : :obj:`str` or None, default='fpr'
            :term:`False positive control<FPR correction>` meaning of cluster
            forming threshold: 'fpr', 'fdr', 'bonferroni' or
            ``None``.

        min_distance : :obj:`float`, default=8.0
            For display purposes only.
            Minimum distance between subpeaks in mm.

        two_sided : :obj:`bool`, default=False
            Whether to employ two-sided thresholding
            or to evaluate positive values only.

        plot_type : {'slice', 'glass'}, default='slice'
            Specifies the type of plot to be drawn for the statistical maps.

        %(cut_coords)s

        display_mode : {'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz', 'l', 'r',\
        'lr', 'lzr', 'lyr', 'lzry', 'lyrz'}, optional
            Choose the direction of the cuts:

            - 'x' - sagittal
            - 'y' - coronal
            - 'z' - axial
            - 'l' - sagittal left hemisphere only
            - 'r' - sagittal right hemisphere only
            - 'ortho' - three cuts are performed in orthogonal directions

            Default is 'z' if ``plot_type`` is 'slice'; 'ortho' if
            ``plot_type`` is 'glass'.

        report_dims : Sequence[ :obj:`int`, :obj:`int` ], default=(1600, 800)
            Specifies width, height (in pixels) of report window
            within a notebook.
            Only applicable when inserting the report into a Jupyter notebook.
            Can be set after report creation using ``report.width``,
            ``report.height``.

        Returns
        -------
        report_text : :class:`~nilearn.reporting.HTMLReport`
            Contains the HTML code for the :term:`GLM` report.

        """
        from nilearn.reporting.glm_reporter import make_glm_report

        if not hasattr(self, "_reporting_data"):
            self._reporting_data = {
                "trial_types": [],
                "noise_model": getattr(self, "noise_model", None),
                "hrf_model": getattr(self, "hrf_model", None),
                "drift_model": None,
            }

        return make_glm_report(
            self,
            contrasts,
            title=title,
            bg_img=bg_img,
            threshold=threshold,
            alpha=alpha,
            cluster_threshold=cluster_threshold,
            height_control=height_control,
            two_sided=two_sided,
            min_distance=min_distance,
            plot_type=plot_type,
            cut_coords=cut_coords,
            display_mode=display_mode,
            report_dims=report_dims,
            input=input,
            verbose=verbose,
        )
