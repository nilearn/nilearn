import pytest


def test_import_from_input_data_with_warning():
    """Tests that importing maskers from deprecated module ``input_data``
    still works.
    """
    from nilearn import input_data, maskers

    assert input_data != maskers
    assert maskers.NiftiMasker == input_data.NiftiMasker
    assert maskers.NiftiLabelsMasker == input_data.NiftiLabelsMasker
    assert maskers.NiftiMapsMasker == input_data.NiftiMapsMasker
    assert maskers.NiftiSpheresMasker == input_data.NiftiSpheresMasker
    assert maskers.MultiNiftiMasker == input_data.MultiNiftiMasker
    from nilearn.input_data import NiftiMapsMasker as masker1
    from nilearn.input_data.nifti_maps_masker import NiftiMapsMasker as masker2

    assert masker1 is masker2
    assert masker1.__module__ == "nilearn.maskers.nifti_maps_masker"
    assert masker2.__module__ == "nilearn.maskers.nifti_maps_masker"
    # Importing privates doesn't work
    with pytest.raises(ImportError):
        from nilearn.input_data.nifti_masker import _filter_and_mask  # noqa
    from nilearn.maskers.nifti_masker import _filter_and_mask  # noqa
