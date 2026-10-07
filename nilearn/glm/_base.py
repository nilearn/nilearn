import itertools
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from nibabel.onetime import auto_attr
from sklearn.utils import Bunch
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._base import NilearnBaseEstimator
from nilearn._utils.cache_mixin import CacheMixin
from nilearn._utils.glm import coerce_to_dict
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.logger import find_stack_level
from nilearn._utils.param_validation import check_is_of_allowed_type
from nilearn.glm._reporting_utils import (
    GLMReportMixin,
    get_runwise_dict,
    make_stat_maps_contrast_clusters,
    mask_to_plot,
    sanitize_generate_report_input,
    turn_into_full_path,
)
from nilearn.image import check_niimg
from nilearn.interfaces.bids.utils import bids_entities, create_bids_filename
from nilearn.maskers import (
    NiftiLabelsMasker,
    NiftiSpheresMasker,
    SurfaceLabelsMasker,
    SurfaceMasker,
)
from nilearn.nilearn_typing import NiimgLike
from nilearn.surface import SurfaceImage
from nilearn.utils.tags import InputTags

FIGURE_FORMAT = "png"


class BaseGLM(GLMReportMixin, CacheMixin, NilearnBaseEstimator):
    """Implement a base class \
    for the :term:`General Linear Model<GLM>`.
    """

    _estimator_type = "glm"  # TODO (sklearn >= 1.8) remove

    def _doc_link_url_param_generator(self, *args):  # noqa : ARG002
        """Return doc URL components for GLM estimators.

        GLM doc URL is slightly different than that of other estimators.

        # TODO (sklearn >= 1.7) remove *args from signature
        """
        estimator_name = self.__class__.__name__
        tmp = list(
            itertools.takewhile(
                lambda part: not part.startswith("_"),
                self.__class__.__module__.split("."),
            )
        )
        estimator_module = ".".join([tmp[0], tmp[1], tmp[2]])
        return {
            "estimator_module": estimator_module,
            "estimator_name": estimator_name,
        }

    def _is_volume_glm(self) -> bool:
        """Return if model is run on volume data or not."""
        return not (
            (
                hasattr(self, "mask_img")
                and isinstance(self.mask_img, (SurfaceMasker, SurfaceImage))
            )
            or (
                self.__sklearn_is_fitted__()
                and hasattr(self, "masker_")
                and isinstance(self.masker_, SurfaceMasker)
            )
        )

    def _is_first_level_glm(self) -> bool:
        """Return True if this estimator is of type FirstLevelModel; False
        otherwise.
        """
        return False

    @property
    def _mask_img(self) -> Nifti1Image | SurfaceImage | None:
        """Return mask image using during fit or mask image passed at init."""
        if self.__sklearn_is_fitted__():
            return self.mask_img_
        if self.mask_img is None:
            return None
        try:
            # load mask_img if is a niiimg-like object
            return check_niimg(self.mask_img)
        except Exception:
            return self.mask_img

    @property
    def mask_img_(self) -> Nifti1Image | SurfaceImage:
        """Return mask image using during fit."""
        check_is_fitted(self)
        return self.masker_.mask_img_

    def _attributes_to_dict(self) -> dict[str, Any]:
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
        if self.__str__() == "First Level Model":
            if self.hrf_model == "fir":
                selected_attributes.append("fir_delays")

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

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=True)
        tags.estimator_type = "glm"
        return tags

    # @auto_attr store the value as an object attribute after initial call
    # better performance than @property
    @auto_attr
    def residuals_(self):
        """Transform element-wise residuals to the same shape \
        as the input image.

        Returns
        -------
        list[Nifti1Image] or list[SurfaceImage]

        """
        return self._get_element_wise_model_attribute(
            "residuals", result_as_time_series=True
        )

    @auto_attr
    def residuals(self):
        """Transform element-wise residuals to the same shape \
        as the input image.

        .. nilearn_deprecated:: 0.14.0

        """
        # TODO (nilearn>=0.16.0) remove the method
        warnings.warn(
            stacklevel=find_stack_level(),
            category=FutureWarning,
            message=(
                "residuals' is deprecated.\n "
                "It will be removed in Nilearn 0.16.0.\n"
                "Use 'residuals_' instead."
            ),
        )
        return self.residuals_

    # @auto_attr store the value as an object attribute after initial call
    # better performance than @property
    @auto_attr
    def predicted_(self):
        """Transform element-wise predicted values to the same shape \
        as the input image.

        Returns
        -------
        list[Nifti1Image] or list[SurfaceImage]

        """
        return self._get_element_wise_model_attribute(
            "predicted", result_as_time_series=True
        )

    @auto_attr
    def predicted(self):
        """Transform element-wise predicted to the same shape \
        as the input image.

        .. nilearn_deprecated:: 0.14.0

        """
        # TODO (nilearn>=0.16.0) remove the method
        warnings.warn(
            stacklevel=find_stack_level(),
            category=FutureWarning,
            message=(
                "residuals' is deprecated.\n "
                "It will be removed in Nilearn 0.16.0.\n"
                "Use 'residuals_' instead."
            ),
        )
        return self.predicted_

    # @auto_attr store the value as an object attribute after initial call
    # better performance than @property
    @auto_attr
    def r_square_(self):
        """Transform element-wise r-squared values to the same shape \
        as the input image.

        Returns
        -------
        list[Nifti1Image] or list[SurfaceImage]

        """
        return self._get_element_wise_model_attribute(
            "r_square", result_as_time_series=False
        )

    @auto_attr
    def r_square(self):
        """Transform element-wise r-squared to the same shape \
        as the input image.

        .. nilearn_deprecated:: 0.14.0

        """
        # TODO (nilearn>=0.16.0) remove the method
        warnings.warn(
            stacklevel=find_stack_level(),
            category=FutureWarning,
            message=(
                "residuals' is deprecated.\n "
                "It will be removed in Nilearn 0.16.0.\n"
                "Use 'residuals_' instead."
            ),
        )
        return self.r_square_

    def _generate_filenames_output(
        self, prefix, contrasts, contrast_types, out_dir, entities_to_drop=None
    ):
        """Generate output filenames for a series of contrasts.

        This function constructs and stores the expected output filenames
        for contrast-related statistical maps and design matrices within
        the model.

        Output files try to follow the BIDS convention where applicable.
        For first level models,
        if no prefix is passed,
        and str or Path were used as input files to the GLM
        the output filenames will be based on the input files.

        See nilearn.glm.io.save_glm_to_bids for more details.

        Parameters
        ----------
        prefix : :obj:`str`
            String to prepend to generated filenames.
            If a string is provided, '_' will be added to the end.

        contrasts : :obj:`str` or array of shape (n_col) or :obj:`list` \
                of (:obj:`str` or array of shape (n_col)) or :obj:`dict`
                Contrast definitions.

        contrast_types : :obj:`dict` of :obj:`str`
            An optional dictionary mapping some
            or all of the :term:`contrast` names to
            specific contrast types ('t' or 'F').

        out_dir : :obj:`str` or :obj:`pathlib.Path`
            Output directory for files.

        entities_to_drop : :obj:`list` of :obj:`str` or None, default=None
                           name of BIDS entities to drop
                           from input filenames
                           when generating output filenames.
                           If None is passed this will default to:
                           ["part", "echo", "hemi", "desc"]

        Notes
        -----
        - The function ensures that contrast names are valid strings.
        - It constructs filenames for effect sizes, statistical maps,
          and design matrices in a structured manner.
        - The output directory structure may include a subject-level
          or group-level subdirectory based on the model type.
        """
        check_is_fitted(self)

        generate_bids_name = _use_input_files_for_filenaming(self, prefix)

        contrasts = coerce_to_dict(contrasts)
        for k, v in contrasts.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"contrast names must be strings, not {type(k)}"
                )

            if not isinstance(v, (str, np.ndarray, list)):
                raise TypeError(
                    "contrast definitions must be strings or array_likes, "
                    f"not {v.__class__.__name__}"
                )

        entities = {"sub": None, "ses": None, "task": None, "space": None}

        if generate_bids_name:
            # try to figure out filename entities from input files
            # only keep entity label if unique across runs
            for k in entities:
                label = [
                    x["entities"].get(k)
                    for x in self._reporting_data["run_imgs"].values()
                    if x["entities"].get(k) is not None
                ]

                label = set(label)
                if len(label) != 1:
                    continue
                label = next(iter(label))
                entities[k] = label

        elif not isinstance(prefix, str):
            prefix = ""

        if self.__str__() == "Second Level Model":
            sub = "group"
        elif entities["sub"]:
            sub = f"sub-{entities['sub']}"
        else:
            sub = prefix.split("_")[0] if prefix.startswith("sub-") else ""

        if self.__str__() == "Second Level Model":
            design_matrices = [self.design_matrix_]
        else:
            design_matrices = self.design_matrices_

        # dropping some entities to avoid polluting output names
        all_entities = [
            *bids_entities()["raw"],
            *bids_entities()["derivatives"],
        ]
        if entities_to_drop is None:
            entities_to_drop = ["part", "echo", "hemi", "desc"]
        assert all(isinstance(x, str) for x in entities_to_drop)
        entities_to_include = [
            x for x in all_entities if x not in entities_to_drop
        ]
        if not generate_bids_name:
            entities_to_include = ["run"]
        entities_to_include.extend(["contrast", "stat"])

        mask = _generate_mask(
            self, prefix, generate_bids_name, entities, entities_to_include
        )

        statistical_maps = _generate_statistical_maps(
            self,
            prefix,
            contrasts,
            contrast_types,
            generate_bids_name,
            entities,
            entities_to_include,
        )

        model_level_mapping = _generate_model_level_mapping(
            self,
            prefix,
            design_matrices,
            generate_bids_name,
            entities,
            entities_to_include,
        )

        design_matrices_dict = _generate_design_matrices_dict(
            self,
            prefix,
            design_matrices,
            generate_bids_name,
            entities_to_include,
        )

        contrasts_dict = _generate_contrasts_dict(
            self,
            prefix,
            contrasts,
            design_matrices,
            generate_bids_name,
            entities,
            entities_to_include,
        )

        out_dir = Path(out_dir) / sub

        # consider using a class or data class
        # to better standardize naming
        self._reporting_data["filenames"] = {
            "dir": out_dir,
            "use_absolute_path": False,
            "mask": mask,
            "design_matrices_dict": design_matrices_dict,
            "contrasts_dict": contrasts_dict,
            "statistical_maps": statistical_maps,
            "model_level_mapping": model_level_mapping,
        }

    def _get_masker_info(self):
        masker_info = {}

        if self.__sklearn_is_fitted__():
            masker_info["n_elements"] = self.masker_._report_content[
                "n_elements"
            ]
            masker_info["coverage"] = (
                f"{self.masker_._report_content['coverage']:0.1f}"
            )

        return masker_info

    def _generate_report_content(
        self,
        contrasts,
        bg_img,
        first_level_contrast,
        threshold,
        alpha,
        cluster_threshold,
        height_control,
        two_sided,
        min_distance,
        cut_coords,
        display_mode,
        plot_type,
    ):
        threshold, cut_coords, first_level_contrast, warning_messages = (
            sanitize_generate_report_input(
                height_control,
                threshold,
                cut_coords,
                plot_type,
                first_level_contrast,
                self,
            )
        )
        for message in warning_messages:
            self._append_report_warning(message)
        contrasts = coerce_to_dict(contrasts)

        # If some contrasts are passed
        # we do not rely on filenames stored in the model.
        output = None
        if contrasts is None:
            output = self._reporting_data.get("filenames")
            if output is not None and output.get("use_absolute_path", True):
                output = turn_into_full_path(output, output["dir"])

            self._append_report_warning(
                "No contrast passed during report generation."
            )

        bg_img = self._load_bg_img(bg_img, self._is_volume_glm())
        self._report_content["mask_plot"] = mask_to_plot(self, bg_img)
        self._report_content.update(self._get_masker_info())
        self._report_content["results"] = make_stat_maps_contrast_clusters(
            model=self,
            contrasts=contrasts,
            output=output,
            first_level_contrast=first_level_contrast,
            threshold_orig=threshold,
            alpha=alpha,
            cluster_threshold=cluster_threshold,
            height_control=height_control,
            two_sided=two_sided,
            min_distance=min_distance,
            bg_img=bg_img,
            cut_coords=cut_coords,
            display_mode=display_mode,
            plot_type=plot_type,
        )
        self._report_content["run_wise_dict"] = self._get_runwise_dict(
            contrasts, output
        )
        # for methods writing, only keep the contrast expressed as strings
        if contrasts is not None:
            contrasts = [x for x in contrasts.values() if isinstance(x, str)]

        self._report_content["contrasts"] = contrasts
        self._report_content["reporting_data"] = Bunch(**self._reporting_data)

    def _get_report_statistical_maps(
        self, contrasts, output, first_level_contrast=None
    ):

        statistical_maps = {}
        if self._is_volume_glm() and output is not None:
            try:
                statistical_maps = {
                    contrast_name: output["dir"]
                    / output["statistical_maps"][contrast_name]["z_score"]
                    for contrast_name in output["statistical_maps"]
                }
            except KeyError:  # pragma: no cover
                if contrasts is not None:
                    statistical_maps = self._make_stat_maps(
                        contrasts,
                        output_type="z_score",
                        first_level_contrast=first_level_contrast,
                    )
        elif contrasts is not None:
            statistical_maps = self._make_stat_maps(
                contrasts,
                output_type="z_score",
                first_level_contrast=first_level_contrast,
            )

        return statistical_maps

    def _get_runwise_dict(self, contrasts, output):
        design_matrices = None

        if self.__sklearn_is_fitted__():
            design_matrices = (
                [self.design_matrix_]
                if self.__str__() == "Second Level Model"
                else self.design_matrices_
            )

        return get_runwise_dict(contrasts, output, design_matrices)

    def _plotting_pred_and_res(
        self,
        observed_ts,
        predicted_ts,
        residuals_ts,
        title_ref: str | None = None,
        figsize: tuple[int, int] = (10, 8),
        close: bool = True,
    ):
        """Help plot observed vs predicted signal and residuals.

        Parameters
        ----------
        observed_ts : array-like
            The observed time series.
        predicted_ts : array-like
            The predicted time series.
        residuals_ts : array-like
            The residuals time series.
        title_ref : str or None, default = None
            Reference string for the title of the plots.
        figsize : tuple of int, default = (10,8)
            Size of the figure.
        close : bool, default = True
            Whether to close the figure after creation.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.
        """
        import matplotlib.pyplot as plt

        max_abs_residual = np.max(np.abs(residuals_ts))
        if max_abs_residual == 0:
            max_abs_residual = 1

        if self.__str__() == "Second Level Model":
            fig, axes = plt.subplots(2, 1, figsize=figsize)

            ax_for_resdiduals_hist = 1

            vmin = np.min([np.min(observed_ts), np.min(predicted_ts)])
            vmax = np.max([np.max(observed_ts), np.max(predicted_ts)])

            # Plot observed vs predicted signal
            axes[0].scatter(observed_ts, predicted_ts, color="blue")
            axes[0].plot(
                [vmin, vmax],
                [vmin, vmax],
                color="black",
                linestyle="--",
                alpha=0.7,
            )
            axes[0].set_title("Observed vs Predicted Signal")
            axes[0].set_ylabel("Observed (AU)")
            axes[0].set_xlabel("Predicted Signal (AU)")

        else:
            fig, axes = plt.subplots(3, 1, figsize=figsize)

            ax_for_resdiduals_hist = 2

            # Generate a time axis
            n_timepoints = len(observed_ts)
            time_axis = np.arange(n_timepoints)
            x_label = "Time (TR)"

            # Plot observed vs predicted signal
            axes[0].plot(
                time_axis, observed_ts, label="Observed", color="blue"
            )
            axes[0].plot(
                time_axis, predicted_ts, label="Predicted", color="orange"
            )
            axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.7)
            axes[0].set_title("Observed vs Predicted Signal")
            axes[0].set_ylabel("Signal Intensity (AU)")
            axes[0].legend()
            axes[0].set_xlabel(x_label)

            # Plot residuals
            axes[1].plot(
                time_axis, residuals_ts, label="Residuals", color="red"
            )
            axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.7)
            axes[1].set_title("Residuals Over Time")
            axes[1].set_ylabel("Residuals")
            axes[1].set_xlabel(x_label)
            axes[1].set_ylim(max_abs_residual * -1.1, max_abs_residual * 1.1)
            axes[1].legend()

        # Plot histogram of residuals
        axes[ax_for_resdiduals_hist].hist(
            residuals_ts, bins=30, color="green", alpha=0.7
        )
        axes[ax_for_resdiduals_hist].set_title("Histogram of Residuals")
        axes[ax_for_resdiduals_hist].set_xlabel("Residuals")
        axes[ax_for_resdiduals_hist].set_ylabel("Frequency")
        axes[ax_for_resdiduals_hist].set_xlim(
            max_abs_residual * -1.1, max_abs_residual * 1.1
        )

        if title_ref is not None:
            fig.suptitle(f"{title_ref}", fontsize=16)

        plt.tight_layout()
        if close:
            plt.close(fig)
        return fig

    def _get_predicted_signal_and_residuals(
        self, coords=None, mask=None, radius: float = 3.0
    ) -> tuple[list[pd.DataFrame], list[str]]:
        """Return observed, predicted and residuals as a list of DataFrames.

        Parameters
        ----------
        coords : :obj:`tuple` or :obj:`list` of :obj:`tuple` of coordinates, \
                or None, default = None
            Coordinates of the voxel(s) or region center(s).
            Ignored if ``masker`` is provided.

        mask : A Niimg-like, :class:`~nilearn.surface.SurfaceImage`, \
               class:`~maskers.NiftiSpheresMasker`,  \
               class:`~maskers.NiftiLabelsMasker`,  \
               class:`~maskers.SurfaceLabelsMasker`,  \
               or None, default = None
            If a SurfaceImage is passed, it will be used
            to instantiate a SurfaceLabelsMasker.
            If a Niimglike is passed, it will be used
            to instantiate a NiftiLabelsMasker.
            If None, a NiftiSpheresMasker
            centered on ``coords`` with radius ``radius`` is created.

        radius : :obj:`float`, default = 3.0
            Radius of the sphere if ``masker`` is None.

        Returns
        -------
        dfs : :obj:`list` of :class:`pandas.DataFrame`
            List of DataFrames containing the observed, predicted,
            and residuals.
            Each dataframe corresponds to a different region.

        region_names : :obj:`list` of :obj:`str`
            List of the region names.
        """
        check_is_fitted(self)

        if self.minimize_memory:
            raise ValueError(
                "To plot predicted signal and residuals, "
                "the GLM model object needs to store "
                "there attributes. "
                "To do so, set 'minimize_memory' to 'False' "
                "when initializing the GLM model."
            )

        if mask is not None and coords is not None:
            warnings.warn(
                (
                    "You provided both 'mask' and 'coords'. "
                    "Only 'mask' will be used."
                ),
                UserWarning,
                stacklevel=find_stack_level(),
            )
            coords = None

        if mask is None:
            if coords is None:
                raise ValueError("Either 'mask' or 'coords' must be provided.")
            # Allow a single coordinate tuple to be passed
            if isinstance(coords[0], (int, float)):
                coords = [coords]
            masker = NiftiSpheresMasker(seeds=coords, radius=radius)
        else:
            check_is_of_allowed_type(
                mask,
                (
                    NiimgLike,
                    SurfaceImage,
                    NiftiSpheresMasker,
                    NiftiLabelsMasker,
                    SurfaceLabelsMasker,
                ),
                "mask",
            )
            if isinstance(mask, NiimgLike):
                if isinstance(self.mask_img_, SurfaceImage):
                    raise TypeError(
                        "The model was fitted with with surface data: "
                        "'mask' must be a SurfaceImage."
                    )
                masker = NiftiLabelsMasker(mask)
            elif isinstance(mask, SurfaceImage):
                if isinstance(self.mask_img_, Nifti1Image):
                    raise TypeError(
                        "The model was fitted with with volume data: "
                        "'mask' must be a NiimgLike."
                    )
                masker = SurfaceLabelsMasker(mask)
            else:
                masker = mask
        if not masker.__sklearn_is_fitted__():
            masker.fit()

        # Get observed, predicted, and residual time series
        # and extract time series for the observed, predicted, and residuals
        y_pred = self.predicted_
        resid = self.residuals_
        if not isinstance(y_pred, (Nifti1Image, SurfaceImage)):
            y_pred = y_pred[0]
            resid = resid[0]
        predicted = masker.transform(y_pred)
        residuals = masker.transform(resid)
        observed = predicted + residuals

        dfs = []
        region_names = masker.get_feature_names_out()

        for i in range(len(region_names)):
            tmp = {}
            tmp[f"{region_names[i]}; observed"] = observed[:, i]
            tmp[f"{region_names[i]}; predicted"] = predicted[:, i]
            tmp[f"{region_names[i]}; residuals"] = residuals[:, i]
            dfs.append(pd.DataFrame(tmp))

        return dfs, region_names

    def plot_predicted_signal_and_residuals(
        self,
        coords=None,
        mask=None,
        radius=3.0,
        figsize=(10, 8),
        show=False,
    ):
        """Plot the predicted and residuals for a small region.

        The model must be fitted.

        Parameters
        ----------
        coords : :obj:`tuple` or :obj:`list` of :obj:`tuple` of coordinates, \
                or None, default = None
            Coordinates of the voxel(s) or region center(s).
            Ignored if ``masker`` is provided.

        mask : A Niimg-like, :class:`~nilearn.surface.SurfaceImage`, \
               class:`~maskers.NiftiSpheresMasker`,  \
               class:`~maskers.NiftiLabelsMasker`,  \
               class:`~maskers.SurfaceLabelsMasker`,  \
               or None, default = None
            If a SurfaceImage is passed, it will be used
            to instantiate a SurfaceLabelsMasker.
            If a Niimglike is passed, it will be used
            to instantiate a NiftiLabelsMasker.
            If None, a NiftiSpheresMasker
            centered on ``coords`` with radius ``radius`` is created.

        radius : :obj:`float`, default = 3.0
            Radius of the sphere if ``masker`` is None.

        figsize : :obj:`tuple`, default = (10, 8)
            Size of the figure.

        show : :obj:`bool`, default = False
            Whether to display the figure.

        Returns
        -------
        dfs : :obj:`list` of :class:`pandas.DataFrame`
            List of DataFrames containing the observed, predicted,
            and residuals values.
            Each dataframe corresponds to a different region.

        fig : list of matplotlib.figure.Figure or None
            The generated figures.

        Notes
        -----
        This method requires that the model was fitted with
        ``minimize_memory=False``, since the voxelwise predicted signal
        and residuals are only stored in that mode.

        """
        dfs, region_names = self._get_predicted_signal_and_residuals(
            coords=coords, mask=mask, radius=radius
        )
        if not is_matplotlib_installed():
            warnings.warn(
                "Matplotlib is not installed. No figure will be returned.",
                ImportWarning,
                stacklevel=find_stack_level(),
            )
            return dfs, None

        figs = []
        for df, region_name in zip(dfs, region_names, strict=False):
            fig = self._plotting_pred_and_res(
                observed_ts=df[f"{region_name}; observed"].values,
                predicted_ts=df[f"{region_name}; predicted"].values,
                residuals_ts=df[f"{region_name}; residuals"].values,
                title_ref=region_name if len(dfs) > 1 else None,
                figsize=figsize,
                close=not show,
            )
            if show:
                fig.show()
            figs.append(fig)

        return dfs, figs


def _generate_mask(
    model,
    prefix: str,
    generate_bids_name: bool,
    entities,
    entities_to_include: list[str],
):
    """Return filename for GLM mask."""
    extension = "gii"
    if model._is_volume_glm():
        extension = "nii.gz"
    fields = {
        "prefix": prefix,
        "suffix": "mask",
        "extension": extension,
        "entities": deepcopy(entities),
    }
    fields["entities"].pop("run", None)
    fields["entities"].pop("ses", None)

    if generate_bids_name:
        fields["prefix"] = ""

    return create_bids_filename(fields, entities_to_include)


def _generate_statistical_maps(
    model,
    prefix: str,
    contrasts,
    contrast_types,
    generate_bids_name: bool,
    entities,
    entities_to_include: list[str],
):
    """Return dictionary containing statmap filenames for each contrast.

    statistical_maps[contrast_name][statmap_label] = filename
    """
    extension = "gii"
    if model._is_volume_glm():
        extension = "nii.gz"

    if not isinstance(contrast_types, dict):
        contrast_types = {}

    statistical_maps: dict[str, dict[str, str]] = {}

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

        fields = {
            "prefix": prefix,
            "suffix": "statmap",
            "extension": extension,
            "entities": deepcopy(entities),
        }

        if generate_bids_name:
            fields["prefix"] = ""

        fields["entities"]["contrast"] = _clean_contrast_name(contrast_name)

        tmp = {}
        for key, stat_label in zip(
            [
                "effect_size",
                "stat",
                "effect_variance",
                "z_score",
                "p_value",
            ],
            ["effect", stat_type, "variance", "z", "p"],
            strict=False,
        ):
            fields["entities"]["stat"] = stat_label
            tmp[key] = create_bids_filename(fields, entities_to_include)

        fields["entities"]["stat"] = None
        fields["suffix"] = "clusters"
        fields["extension"] = "tsv"
        tmp["clusters_tsv"] = create_bids_filename(fields, entities_to_include)

        fields["extension"] = "json"
        tmp["metadata"] = create_bids_filename(fields, entities_to_include)

        statistical_maps[contrast_name] = Bunch(**tmp)

    return statistical_maps


def _generate_model_level_mapping(
    model,
    prefix: str,
    design_matrices,
    generate_bids_name: bool,
    entities,
    entities_to_include: list[str],
):
    """Return dictionary of filenames for nifti of runwise error & residuals.

    model_level_mapping[i_run][statmap_label] = filename
    """
    extension = "gii"
    if model._is_volume_glm():
        extension = "nii.gz"
    fields = {
        "prefix": prefix,
        "suffix": "statmap",
        "extension": extension,
        "entities": deepcopy(entities),
    }

    if generate_bids_name:
        fields["prefix"] = ""

    model_level_mapping = {}

    for i_run, _ in enumerate(design_matrices):
        if _is_flm_with_single_run(model):
            fields["entities"]["run"] = i_run + 1
        if generate_bids_name:
            fields["entities"] = deepcopy(
                model._reporting_data["run_imgs"][i_run]["entities"]
            )

        tmp = {}
        for key, stat_label in zip(
            ["residuals", "r_square"],
            ["errorts", "rsquared"],
            strict=False,
        ):
            fields["entities"]["stat"] = stat_label
            tmp[key] = create_bids_filename(fields, entities_to_include)

        model_level_mapping[i_run] = Bunch(**tmp)

    return model_level_mapping


def _generate_design_matrices_dict(
    model,
    prefix: str,
    design_matrices,
    generate_bids_name: bool,
    entities_to_include: list[str],
) -> dict[int, dict[str, str]]:
    """Return dictionary with filenames for design_matrices figures / tables.

    design_matrices_dict[i_run][key] = filename
    """
    fields = {"prefix": prefix, "extension": FIGURE_FORMAT, "entities": {}}
    if generate_bids_name:
        fields["prefix"] = None  # type: ignore[assignment]

    design_matrices_dict = Bunch()

    for i_run, _ in enumerate(design_matrices):
        if _is_flm_with_single_run(model):
            fields["entities"] = {"run": i_run + 1}  # type: ignore[assignment]
        if generate_bids_name:
            fields["entities"] = deepcopy(
                model._reporting_data["run_imgs"][i_run]["entities"]
            )

        tmp = {}
        for extension in [FIGURE_FORMAT, "tsv"]:
            for key, suffix in zip(
                ["design_matrix", "correlation_matrix"],
                ["design", "corrdesign"],
                strict=False,
            ):
                fields["extension"] = extension
                fields["suffix"] = suffix
                tmp[f"{key}_{extension}"] = create_bids_filename(
                    fields, entities_to_include
                )

        design_matrices_dict[i_run] = Bunch(**tmp)

    return design_matrices_dict


def _generate_contrasts_dict(
    model,
    prefix: str,
    contrasts,
    design_matrices,
    generate_bids_name: bool,
    entities,
    entities_to_include: list[str],
) -> dict[int, dict[str, str]]:
    """Return dictionary with filenames for contrast matrices figures.

    contrasts_dict[i_run][contrast_name] = filename
    """
    fields = {
        "prefix": prefix,
        "extension": FIGURE_FORMAT,
        "entities": deepcopy(entities),
        "suffix": "design",
    }
    if generate_bids_name:
        fields["prefix"] = ""

    contrasts_dict = Bunch()

    for i_run, _ in enumerate(design_matrices):
        if _is_flm_with_single_run(model):
            fields["entities"]["run"] = i_run + 1
        if generate_bids_name:
            fields["entities"] = deepcopy(
                model._reporting_data["run_imgs"][i_run]["entities"]
            )

        tmp = {}
        for contrast_name in contrasts:
            fields["entities"]["contrast"] = _clean_contrast_name(
                contrast_name
            )
            tmp[contrast_name] = create_bids_filename(
                fields, entities_to_include
            )

        contrasts_dict[i_run] = Bunch(**tmp)

    return contrasts_dict


def _use_input_files_for_filenaming(self, prefix) -> bool:
    """Determine if we should try to use input files to generate \
       output filenames.
    """
    if self.__str__() == "Second Level Model" or prefix is not None:
        return False

    input_files = self._reporting_data["run_imgs"]

    files_used_as_input = all(len(x) > 0 for x in input_files.values())
    tmp = {x.get("sub") for x in input_files.values()}
    all_files_have_same_sub = len(tmp) == 1 and tmp is not None

    return files_used_as_input and all_files_have_same_sub


def _is_flm_with_single_run(model) -> bool:
    return (
        model.__str__() == "First Level Model"
        and len(model._reporting_data["run_imgs"]) > 1
    )


def _clean_contrast_name(contrast_name):
    """Remove prohibited characters from name and convert to camelCase.

    .. nilearn_versionadded:: 0.9.2

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
            stacklevel=find_stack_level(),
        )
    return new_name
