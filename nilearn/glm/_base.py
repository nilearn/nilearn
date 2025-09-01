import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
from nibabel.onetime import auto_attr
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_is_fitted

from nilearn._utils.cache_mixin import CacheMixin
from nilearn._utils.glm import coerce_to_dict
from nilearn._utils.logger import find_stack_level
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.externals import tempita
from nilearn.interfaces.bids.utils import bids_entities, create_bids_filename
from nilearn.maskers import SurfaceMasker
from nilearn.surface import SurfaceImage

FIGURE_FORMAT = "png"


class BaseGLM(CacheMixin, BaseEstimator):
    """Implement a base class \
    for the :term:`General Linear Model<GLM>`.
    """

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
                raise ValueError(
                    f"contrast names must be strings, not {type(k)}"
                )

            if not isinstance(v, (str, np.ndarray, list)):
                raise ValueError(
                    "contrast definitions must be strings or array_likes, "
                    f"not {type(v)}"
                )

        entities = {
            "sub": None,
            "ses": None,
            "task": None,
            "space": None,
        }

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
        ):
            fields["entities"]["stat"] = stat_label
            tmp[key] = create_bids_filename(fields, entities_to_include)

        fields["entities"]["stat"] = None
        fields["suffix"] = "clusters"
        fields["extension"] = "tsv"
        tmp["clusters_tsv"] = create_bids_filename(fields, entities_to_include)

        fields["extension"] = "json"
        tmp["metadata"] = create_bids_filename(fields, entities_to_include)

        statistical_maps[contrast_name] = tempita.bunch(**tmp)

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
        ):
            fields["entities"]["stat"] = stat_label
            tmp[key] = create_bids_filename(fields, entities_to_include)

        model_level_mapping[i_run] = tempita.bunch(**tmp)

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

    design_matrices_dict = tempita.bunch()

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
            ):
                fields["extension"] = extension
                fields["suffix"] = suffix
                tmp[f"{key}_{extension}"] = create_bids_filename(
                    fields, entities_to_include
                )

        design_matrices_dict[i_run] = tempita.bunch(**tmp)

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

    contrasts_dict = tempita.bunch()

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

        contrasts_dict[i_run] = tempita.bunch(**tmp)

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
            stacklevel=find_stack_level(),
        )
    return new_name
