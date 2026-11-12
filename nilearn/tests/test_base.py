import inspect
import re
from abc import ABC, abstractmethod

import pytest
from numpydoc.docscrape import NumpyDocString

from nilearn._base import NilearnBaseEstimator


class TestNilearnBaseEstimator(ABC):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.estimator = self.create_instance()

    @pytest.fixture
    def doc(self):
        return NumpyDocString(self.estimator.__doc__)

    @abstractmethod
    def create_instance(self) -> NilearnBaseEstimator:
        raise NotImplementedError()

    @pytest.fixture
    def fitted_estimator(self):
        raise NotImplementedError()

    def test_sklearn_tags(self):
        tags = self.estimator.__sklearn_tags__().input_tags
        assert hasattr(tags, "niimg_like")
        assert tags.niimg_like is False

    def test_verbose(self):
        assert self.estimator.verbose == 0

        signature = dict(
            **inspect.signature(self.estimator.__init__).parameters
        )
        assert signature["verbose"].default == 0

    def test_doc_sections_exist(self, doc):
        for section in ["Parameters", "Attributes"]:
            assert re.search(
                r"^" + section + "\n^-{10}", str(doc), re.MULTILINE
            ), (
                f"Estimator {self.estimator.__class__.__name__} "
                f"has no '{section} section."
            )

    def test_doc_no_duplicate_params(self, doc):
        documented_parameters: dict[str, str] = {
            param.name: param.type for param in doc["Parameters"]
        }
        assert len(documented_parameters) == len(set(documented_parameters))

    def test_doc_no_duplicate_attrs(self, doc):
        documented_attributes: dict[str, str] = {
            attr.name: attr.type for attr in doc["Attributes"]
        }
        assert len(documented_attributes) == len(set(documented_attributes))

    def test_doc_no_undocumented_params(self, doc):
        # check public attributes before fit
        parameters = [
            x for x in self.estimator.__dict__ if not x.startswith("_")
        ]
        documented_parameters: dict[str, str] = {
            param.name: param.type for param in doc["Parameters"]
        }

        undocumented_parameters = list(
            set(parameters) - set(documented_parameters)
        )
        assert not undocumented_parameters

    def test_doc_no_extra_params(self, doc):
        parameters = [
            x for x in self.estimator.__dict__ if not x.startswith("_")
        ]
        documented_parameters: dict[str, str] = {
            param.name: param.type for param in doc["Parameters"]
        }

        # ensure that there are no extra parameters documented in docstring
        extra_parameters = list(
            set(documented_parameters) - set(parameters) - {"kwargs"}
        )
        assert not extra_parameters, (
            "Extra docstring for parameters "
            f"[{', '.join(extra_parameters)}] "
            f"in estimator {self.estimator.__class__.__name__}."
        )

    def test_doc_param_order(self, doc):
        # Attributes should be in same order as in __init__()
        tmp = dict(**inspect.signature(self.estimator.__init__).parameters)

        documented_parameters: dict[str, str] = {
            param.name: param.type for param in doc["Parameters"]
        }

        assert [str(x) for x in documented_parameters] == [
            str(x) for x in tmp
        ], (
            f"Parameters of {self.estimator.__class__.__name__} "
            f"should be in order\n{list(tmp)}. "
            f"Got\n{list(documented_parameters)}"
        )

    def test_doc_attr_order(self, doc):
        documented_attributes: dict[str, str] = {
            attr.name: attr.type for attr in doc["Attributes"]
        }

        # nice to have
        # if possible attributes should be in alphabetical order
        # not always possible as sometimes doc string are composed from
        # nilearn._utils.docs
        assert list(documented_attributes) != sorted(documented_attributes), (
            f"Attributes of {self.estimator.__class__.__name__} "
            f"should be in order {sorted(documented_attributes)}. "
            f"Got {list(documented_attributes)}"
        )

    def test_doc_no_undocumented_attributes_after_fit(
        self, doc, fitted_estimator
    ):
        fitted_attributes = [
            x
            for x in fitted_estimator.__dict__
            if x.endswith("_") and not x.startswith("_")
        ]
        documented_attributes: dict[str, str] = {
            attr.name: attr.type for attr in doc["Attributes"]
        }
        undocumented_attributes = list(
            set(fitted_attributes) - set(documented_attributes)
        )
        assert not undocumented_attributes

    def test_doc_no_extra_attributes_after_fit(self, doc, fitted_estimator):
        fitted_attributes = [
            x
            for x in fitted_estimator.__dict__
            if x.endswith("_") and not x.startswith("_")
        ]
        documented_attributes: dict[str, str] = {
            attr.name: attr.type for attr in doc["Attributes"]
        }
        extra_attributes = list(
            set(documented_attributes) - set(fitted_attributes)
        )

        assert not extra_attributes

    def test_check_doc_link(self):
        estimator_name = self.estimator.__class__.__name__

        modules = self.estimator.__class__.__module__.split(".")

        extra = r"\."
        # if isinstance(estimator, FirstLevelModel):
        #     extra = r"\.first_level\."
        # elif isinstance(estimator, SecondLevelModel):
        #     extra = r"\.second_level\."

        expected_pattern = (
            rf"https://nilearn\.github\.io/"
            rf"(dev|stable|\d+\.\d+\.\d+)/modules/generated/"
            rf"nilearn\.{modules[1]}{extra}{estimator_name}\.html"
        )

        doc_link = self.estimator._get_doc_link()

        assert re.fullmatch(expected_pattern, doc_link), (
            f"Doc link '{doc_link}' does not match expected pattern "
            f"'{expected_pattern}'"
        )
