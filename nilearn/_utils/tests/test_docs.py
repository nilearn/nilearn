"""Test structured dataset descriptions."""

import pandas as pd
import pytest
from sklearn.utils import Bunch

from nilearn._utils.docs import (
    DATASET_DESCRIPTIONS,
    Description,
    _matches_type,
    check_content_types,
    content_to_rst,
    type_to_rst,
)
from nilearn.datasets._utils import (
    PACKAGE_DIRECTORY,
    get_dataset_descr,
)


@pytest.mark.parametrize(
    "type_, expected",
    [
        (str, ":obj:`str`"),
        ("str", "str"),
        (pd.DataFrame, ":class:`pandas.DataFrame`"),
        (list[str], ":obj:`list` of :obj:`str`"),
        (
            list[tuple[str, list[int]]],
            ":obj:`list` of :obj:`tuple` of "
            "(:obj:`str`, :obj:`list` of :obj:`int`)",
        ),
        (str | None, ":obj:`str` or ``None``"),
    ],
)
def test_type_to_rst(type_, expected):
    """Check rendering of types as rst."""
    assert type_to_rst(type_) == expected


@pytest.mark.parametrize(
    "rst_file",
    sorted((PACKAGE_DIRECTORY / "description").glob("*.rst")),
    ids=lambda x: x.stem,
)
def test_description_directives_are_rendered(rst_file):
    """Check all nilearn_dataset_* directives refer to a registered dataset.

    Rendering raises a KeyError otherwise.
    """
    assert ".. nilearn_dataset_" not in get_dataset_descr(rst_file.stem)


@pytest.mark.parametrize("name", DATASET_DESCRIPTIONS)
def test_description_from_registry(name):
    """Check descriptions built from the registry."""
    description = Description.from_registry(name)

    assert description.documentation.endswith(
        f"/modules/description/{name}.html"
    )
    for value in description.content.values():
        assert {"type", "desc"} <= set(value)


@pytest.mark.parametrize(
    "value, type_, expected",
    [
        ("a", str, True),
        (1, str, False),
        ("a", "anything", True),
        (["a", "b"], list[str], True),
        (["a", 1], list[str], False),
        (("a", "b"), list[str], False),
        ([("a", [1, 2])], list[tuple[str, list[int]]], True),
        ([("a", [1, "2"])], list[tuple[str, list[int]]], False),
        ([("a",)], list[tuple[str, list[int]]], False),
        ((1, 2, 3), tuple[int, ...], True),
        ({"a": 1}, dict[str, int], True),
        ({"a": "1"}, dict[str, int], False),
        (None, str | None, True),
        (pd.DataFrame(), pd.DataFrame, True),
    ],
)
def test_matches_type(value, type_, expected):
    """Check type matching of values."""
    assert _matches_type(value, type_) is expected


def test_check_content_types():
    """Check mismatches between data and their described content."""
    content = Bunch(
        maps=Bunch(type=str, desc=""),
        labels=Bunch(type=list[str], desc=""),
        lut=Bunch(type=pd.DataFrame, desc=""),
    )

    assert (
        check_content_types(
            Bunch(maps="foo", labels=["a"], lut=pd.DataFrame()), content
        )
        == []
    )

    errors = check_content_types(Bunch(maps=1, labels=["a"]), content)

    assert len(errors) == 2
    assert "'maps' expected type" in errors[0]
    assert "'lut' is described but missing" in errors[1]


def test_content_to_rst():

    rst = content_to_rst(name="talairach_atlas", indent="   ")

    assert (
        rst[0:60]
        == """
   - ``atlas_type``: :obj:`str`.  Type of atlas. See :term:"""
    )
