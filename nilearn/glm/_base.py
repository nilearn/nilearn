import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
from nibabel.onetime import auto_attr
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils import CacheMixin
from nilearn._utils.glm import coerce_to_dict
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.externals import tempita
from nilearn.maskers import SurfaceMasker
from nilearn.surface import SurfaceImage


class BaseGLM(CacheMixin, BaseEstimator):
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

    def _more_tags(self):
        """Return estimator tags.

        TODO remove when bumping sklearn_version > 1.5
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO
        # get rid of if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(surf_img=True, niimg_like=True, glm=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=True, glm=True)
        return tags

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
        return self._get_element_wise_model_attribute(
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
        return self._get_element_wise_model_attribute(
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
        return self._get_element_wise_model_attribute(
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
            or to evaluate either positive or negative values only.

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
        )

    def _generate_filenames_output(
        self, prefix, contrasts, contrast_types, out_dir
    ):
        """Generate output filenames for a series of contrasts.

        Store the name of the output files in the model.

        See nilearn.interfaces.bids.save_glm_to_bids for more details.

        Parameters
        ----------
        prefix : :obj:`str`
            String to prepend to generated filenames.
            If a string is provided, '_' will be added to the end.

        contrasts : :obj:`str` or array of shape (n_col) or :obj:`list` \
                of (:obj:`str` or array of shape (n_col)) or :obj:`dict`
                Contrast definitions.

        contrast_types ::obj:`dict` of :obj:`str`
            An optional dictionary mapping some
            or all of the :term:`contrast` names to
            specific contrast types ('t' or 'F').

        out_dir : :obj:`str` or :obj:`pathlib.Path`
            Output directory for files.
        """
        check_is_fitted(self)

        if not isinstance(prefix, str):
            prefix = ""
        if prefix and not prefix.endswith("_"):
            prefix += "_"

        contrasts = coerce_to_dict(contrasts)
        for k, v in contrasts.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"contrast names must be strings, not {type(k)}"
                )

            if not isinstance(v, (str, np.ndarray, list)):
                raise ValueError(
                    "contrast definitions must be strings or array_likes, "
                    f"not {type(v)}"
                )

        if self.__str__() == "Second Level Model":
            sub_directory = "group"
            design_matrices = [self.design_matrix_]
        else:
            sub_directory = (
                prefix.split("_")[0] if prefix.startswith("sub-") else ""
            )
            design_matrices = self.design_matrices_

        out_dir = Path(out_dir) / sub_directory

        design_matrices_dict = tempita.bunch()
        contrasts_dict = tempita.bunch()
        for i_run, _ in enumerate(design_matrices, start=1):
            run_str = f"run-{i_run}_" if len(design_matrices) > 1 else ""

            design_matrices_dict[i_run] = tempita.bunch(
                design_matrix=f"{prefix}{run_str}design.svg",
                correlation_matrix=f"{prefix}{run_str}corrdesign.svg",
            )

            tmp = {
                contrast_name: (
                    f"{prefix}{run_str}"
                    f"contrast-{_clean_contrast_name(contrast_name)}"
                    "_design.svg"
                )
                for contrast_name in contrasts
            }
            contrasts_dict[i_run] = tempita.bunch(**tmp)

        if not isinstance(contrast_types, dict):
            contrast_types = {}

        statistical_maps = {}
        for contrast_name in contrasts:
            # Extract stat_type
            contrast_matrix = contrasts[contrast_name]

            # Strings and 1D arrays are assumed to be t-contrasts
            if isinstance(contrast_matrix, str) or (contrast_matrix.ndim == 1):
                stat_type = "t"
            else:
                stat_type = "F"

            # Override automatic detection with explicit type if provided
            stat_type = contrast_types.get(contrast_name, stat_type)

            # Convert the contrast name to camelCase
            contrast_entity = (
                f"contrast-{_clean_contrast_name(contrast_name)}_"
            )
            suffix = "_statmap.nii.gz"
            statistical_maps[contrast_name] = {
                "effect_size": (
                    f"{prefix}{contrast_entity}stat-effect{suffix}"
                ),
                "stat": (f"{prefix}{contrast_entity}stat-{stat_type}{suffix}"),
                "effect_variance": (
                    f"{prefix}{contrast_entity}stat-variance{suffix}"
                ),
                "z_score": (f"{prefix}{contrast_entity}stat-z{suffix}"),
                "p_value": (f"{prefix}{contrast_entity}stat-p{suffix}"),
            }

        self._reporting_data["filenames"] = {
            "dir": out_dir,
            "design_matrices_dict": design_matrices_dict,
            "contrasts_dict": contrasts_dict,
            "statistical_maps": statistical_maps,
        }

        return self


def _clean_contrast_name(contrast_name):
    """Remove prohibited characters from name and convert to camelCase.

    .. versionadded:: 0.9.2

    BIDS filenames, in which the contrast name will appear as a
    contrast-<name> key/value pair, must be alphanumeric strings.

    Parameters
    ----------
    contrast_name : :obj:`str`
        Contrast name to clean.

    Returns
    -------
    new_name : :obj:`str`
        Contrast name converted to alphanumeric-only camelCase.
    """
    new_name = contrast_name[:]

    # Some characters translate to words
    new_name = new_name.replace("-", " Minus ")
    new_name = new_name.replace("+", " Plus ")
    new_name = new_name.replace(">", " Gt ")
    new_name = new_name.replace("<", " Lt ")

    # Others translate to spaces
    new_name = new_name.replace("_", " ")

    # Convert to camelCase
    new_name = new_name.split(" ")
    new_name[0] = new_name[0].lower()
    new_name[1:] = [c.title() for c in new_name[1:]]
    new_name = " ".join(new_name)

    # Remove non-alphanumeric characters
    new_name = "".join(ch for ch in new_name if ch.isalnum())

    # Let users know if the name was changed
    if new_name != contrast_name:
        warnings.warn(
            f'Contrast name "{contrast_name}" changed to "{new_name}"',
            stacklevel=4,
        )
    return new_name
