import datetime
import uuid
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from nibabel.onetime import auto_attr
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils import logger
from nilearn._utils.cache_mixin import CacheMixin
from nilearn._utils.docs import fill_doc
from nilearn._utils.glm import coerce_to_dict
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.logger import find_stack_level
from nilearn._utils.param_validation import check_params
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._version import __version__
from nilearn.glm._reporting_utils import (
    check_generate_report_input,
    glm_model_attributes_to_dataframe,
    load_bg_img,
    make_stat_maps_contrast_clusters,
    mask_to_plot,
    sanitize_generate_report_input,
    turn_into_full_path,
)
from nilearn.interfaces.bids.utils import bids_entities, create_bids_filename
from nilearn.maskers import SurfaceMasker
from nilearn.reporting._utils import dataframe_to_html
from nilearn.reporting.html_report import (
    MISSING_ENGINE_MSG,
    UNFITTED_MSG,
    HTMLReport,
    assemble_report,
    is_notebook,
    return_jinja_env,
)
from nilearn.surface import SurfaceImage
from nilearn.typing import ClusterThreshold, HeightControl

FIGURE_FORMAT = "png"


class BaseGLM(CacheMixin, BaseEstimator):
    """Implement a base class \
    for the :term:`General Linear Model<GLM>`.
    """

    _estimator_type = "glm"  # TODO (sklearn >= 1.8) remove

    def _is_volume_glm(self):
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

    def _is_first_level_glm(self):
        """Return True if this estimator is of type FirstLevelModel; False
        otherwise.
        """
        return False

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

    def _more_tags(self):
        """Return estimator tags.

        TODO (sklearn >= 1.6.0) remove
        """
        return self.__sklearn_tags__()

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(surf_img=True, niimg_like=True, glm=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=True)
        tags.estimator_type = "glm"
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

        contrast_types ::obj:`dict` of :obj:`str`
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

    @fill_doc
    def generate_report(
        self,
        contrasts=None,
        first_level_contrast=None,
        title: str | None = None,
        bg_img="MNI152TEMPLATE",
        threshold: float | int | np.floating | np.integer | None = None,
        alpha=0.001,
        cluster_threshold: ClusterThreshold = 0,
        height_control: HeightControl = "fpr",
        two_sided: bool = False,
        min_distance: float | int | np.floating | np.integer = 8.0,
        plot_type: Literal["slice", "glass"] = "slice",
        cut_coords=None,
        display_mode=None,
        report_dims=(1600, 800),
    ) -> HTMLReport:
        """Return a :class:`~nilearn.reporting.HTMLReport` \
        which shows all important aspects of a fitted :term:`GLM`.

        The :class:`~nilearn.reporting.HTMLReport` can be opened in a
        browser, displayed in a notebook, or saved to disk as a standalone
        HTML file.

        The :term:`GLM` must be fitted and have the computed design
        matrix(ces).

        Parameters
        ----------
        contrasts : :obj:`dict` with :obj:`str` - ndarray key-value pairs \
            or :obj:`str` \
            or :obj:`list` of :obj:`str` \
            or ndarray or \
            :obj:`list` of ndarray, Default=None

            Contrasts information for a first or second level model.

            Example:

                Dict of :term:`contrast` names and coefficients,
                or list of :term:`contrast` names
                or list of :term:`contrast` coefficients
                or :term:`contrast` name
                or :term:`contrast` coefficient

                Each :term:`contrast` name must be a string.
                Each :term:`contrast` coefficient must be a list
                or numpy array of ints.

            Contrasts are passed to ``contrast_def`` for FirstLevelModel
            (:func:`nilearn.glm.first_level.FirstLevelModel.compute_contrast`)
            & second_level_contrast for SecondLevelModel
            (:func:`nilearn.glm.second_level.SecondLevelModel.compute_contrast`)

        %(first_level_contrast)s

            .. nilearn_versionadded:: 0.12.0

        title : :obj:`str` or None, default=None
            If string, represents the web page's title and primary heading,
            model type is sub-heading.
            If None, page titles and headings are autogenerated
            using :term:`contrast` names.

        bg_img : Niimg-like object, default='MNI152TEMPLATE'
            See :ref:`extracting_data`.
            The background image for mask and stat maps to be plotted on upon.
            To turn off background image, just pass "bg_img=None".

        threshold : :obj:`float` or :obj:`int` or None, default=None
            Cluster forming threshold in same scale as `stat_img` (either a
            t-scale or z-scale value). Used only if height_control is None.
            If ``threshold`` is set to None when ``height_control`` is None,
            ``threshold`` will be set to 3.09.

            .. note::

                - When ``two_sided`` is True:

                  ``'threshold'`` cannot be negative.

                  The given value should be within the range of minimum and
                  maximum intensity of the input image.
                  All intensities in the interval ``[-threshold, threshold]``
                  will be set to zero.

                - When ``two_sided`` is False:

                  - If the threshold is negative:

                    It should be greater than the minimum intensity
                    of the input data.
                    All intensities greater than or equal
                    to the specified threshold will be set to zero.
                    All other intensities keep their original values.

                  - If the threshold is positive:

                    It should be less than the maximum intensity
                    of the input data.
                    All intensities less than or equal
                    to the specified threshold will be set to zero.
                    All other intensities keep their original values.

        alpha : :obj:`float`, default=0.001
            Number controlling the thresholding (either a p-value or q-value).
            Its actual meaning depends on the height_control parameter.
            This function translates alpha to a z-scale threshold.

        %(cluster_threshold)s

        height_control :  :obj:`str` or None, default='fpr'
            false positive control meaning of cluster forming
            threshold: 'fpr' or 'fdr' or 'bonferroni' or None.

        two_sided : :obj:`bool`, default=False
            Whether to employ two-sided thresholding or to evaluate positive
            values only.

        min_distance : :obj:`float`, default=8.0
            For display purposes only.
            Minimum distance between subpeaks in mm.

        plot_type : :obj:`str`, {'slice', 'glass'}, default='slice'
            Specifies the type of plot to be drawn for the statistical maps.

        %(cut_coords)s

            .. note::
                ``cut_coords`` will not be used when ``plot_type='glass'``.


        display_mode :  :obj:`str`, default=None
            Default is 'z' if plot_type is 'slice';
            'ortho' if plot_type is 'glass'.

            Choose the direction of the cuts:
            'x' - sagittal, 'y' - coronal, 'z' - axial,
            'l' - sagittal left hemisphere only,
            'r' - sagittal right hemisphere only,
            'ortho' - three cuts are performed in orthogonal directions.

            Possible values are:
            'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
            'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.

        report_dims : Sequence[:obj:`int`, :obj:`int`], default=(1600, 800)
            Specifies width, height (in pixels) of report window within a
            notebook.
            Only applicable when inserting the report into a Jupyter notebook.
            Can be set after report creation using report.width, report.height.


        Returns
        -------
        report_text : :class:`~nilearn.reporting.HTMLReport`
            Contains the HTML code for the :term:`GLM` report.

        """
        check_generate_report_input(
            height_control, cluster_threshold, min_distance, plot_type
        )
        check_params(locals())

        threshold, cut_coords, first_level_contrast, warning_messages = (
            sanitize_generate_report_input(
                height_control,
                threshold,
                cut_coords,
                plot_type,
                first_level_contrast,
                self._is_first_level_glm(),
            )
        )

        model_attributes = glm_model_attributes_to_dataframe(self)
        with pd.option_context("display.max_colwidth", 100):
            model_attributes_html = dataframe_to_html(
                model_attributes,
                precision=2,
                header=True,
                sparsify=False,
            )

        if not hasattr(self, "_reporting_data"):
            self._reporting_data: dict[str, Any] = {
                "trial_types": [],
                "noise_model": getattr(self, "noise_model", None),
                "hrf_model": getattr(self, "hrf_model", None),
                "drift_model": None,
            }
        contrasts = coerce_to_dict(contrasts)

        # If some contrasts are passed
        # we do not rely on filenames stored in the model.
        output = None
        if contrasts is None:
            output = self._reporting_data.get("filenames", None)
            if output is not None and output.get("use_absolute_path", True):
                output = turn_into_full_path(output, output["dir"])

            warning_messages.append(
                "No contrast passed during report generation."
            )

        design_matrices = None
        mask_plot = None
        mask_info = {"n_elements": 0, "coverage": "0"}
        results = None

        if not self.__sklearn_is_fitted__():
            warning_messages.append(UNFITTED_MSG)

        else:
            design_matrices = (
                [self.design_matrix_]
                if self.__str__() == "Second Level Model"
                else self.design_matrices_
            )

            bg_img = load_bg_img(bg_img, self._is_volume_glm())
            mask_plot = mask_to_plot(self, bg_img)

            # We try to rely on the content of glm object only
            # by reading images from disk rarther than recomputing them
            mask_info = {
                k: v
                for k, v in self.masker_._report_content.items()
                if k in ["n_elements", "coverage"]
            }
            if "coverage" in mask_info:
                mask_info["coverage"] = f"{mask_info['coverage']:0.1f}"

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

            logger.log(
                "Generating contrast-level figures...", verbose=self.verbose
            )
            results = make_stat_maps_contrast_clusters(
                stat_img=statistical_maps,
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

        design_matrices_dict = Bunch()
        contrasts_dict = Bunch()
        if output is not None:
            design_matrices_dict = output["design_matrices_dict"]
            contrasts_dict = output["contrasts_dict"]

        if is_matplotlib_installed():
            from nilearn._utils.plotting import (
                generate_contrast_matrices_figures,
                generate_design_matrices_figures,
            )

            logger.log(
                "Generating design matrices figures...", verbose=self.verbose
            )
            design_matrices_dict = generate_design_matrices_figures(
                design_matrices,
                design_matrices_dict=design_matrices_dict,
                output=output,
            )

            logger.log(
                "Generating contrast matrices figures...", verbose=self.verbose
            )
            contrasts_dict = generate_contrast_matrices_figures(
                design_matrices,
                contrasts,
                contrasts_dict=contrasts_dict,
                output=output,
            )
        else:
            warning_messages.append(MISSING_ENGINE_MSG)

        run_wise_dict = Bunch()
        for i_run in design_matrices_dict:
            tmp = Bunch()
            tmp["design_matrix_png"] = design_matrices_dict[i_run][
                "design_matrix_png"
            ]
            tmp["correlation_matrix_png"] = design_matrices_dict[i_run][
                "correlation_matrix_png"
            ]
            tmp["all_contrasts"] = None
            if i_run in contrasts_dict:
                tmp["all_contrasts"] = contrasts_dict[i_run]
            run_wise_dict[i_run] = tmp

        # for methods writing, only keep the contrast expressed as strings
        if contrasts is not None:
            contrasts = [x for x in contrasts.values() if isinstance(x, str)]

        title = f"<br>{title}" if title else ""
        title = f"Statistical Report - {self.__str__()}{title}"

        smoothing_fwhm = getattr(self, "smoothing_fwhm", None)
        smoothing_fwhm = None if smoothing_fwhm == 0 else smoothing_fwhm

        for msg in warning_messages:
            warnings.warn(
                msg,
                stacklevel=find_stack_level(),
            )

        env = return_jinja_env()

        body_tpl = env.get_template("html/glm/body_glm.jinja")

        # TODO clean up docstring from RST formatting
        docstring = (
            self.__doc__.split("Parameters\n")[0]
            if self.__doc__ is not None
            else ""
        )

        body = body_tpl.render(
            docstring=docstring,
            contrasts=contrasts,
            date=datetime.datetime.now().replace(microsecond=0).isoformat(),
            mask_plot=mask_plot,
            model_type=self.__str__(),
            parameters=model_attributes_html,
            reporting_data=Bunch(**self._reporting_data),
            results=results,
            run_wise_dict=run_wise_dict,
            is_notebook=is_notebook(),
            smoothing_fwhm=smoothing_fwhm,
            title=title,
            version=__version__,
            unique_id=str(uuid.uuid4()).replace("-", ""),
            warning_messages=warning_messages,
            has_plotting_engine=is_matplotlib_installed(),
            **mask_info,
        )

        report = assemble_report(body, title)

        report.resize(*report_dims)

        return report


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
        fields["prefix"] = None

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
            fields["prefix"] = None

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
        fields["prefix"] = None

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
        fields["prefix"] = None

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
