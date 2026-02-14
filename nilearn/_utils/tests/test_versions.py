import pytest

from nilearn._utils.versions import compare_version


@pytest.mark.parametrize(
    "version_a,operator,version_b",
    [
        ("0.1.0", ">", "0.0.1"),
        ("0.1.0", ">=", "0.0.1"),
        ("0.1", "==", "0.1.0"),
        ("0.0.0", "<", "0.1.0"),
        ("1.0", "!=", "0.1.0"),
    ],
)
def test_compare_version(version_a, operator, version_b):
    assert compare_version(version_a, operator, version_b)


def test_compare_version_error():
    with pytest.raises(
        ValueError,
        match=r"'compare_version' received an unexpected operator <>.",
    ):
        compare_version("0.1.0", "<>", "1.1.0")
