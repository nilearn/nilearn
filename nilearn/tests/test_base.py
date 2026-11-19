import inspect
import re
from abc import ABC, abstractmethod

import pytest
from numpydoc.docscrape import NumpyDocString
import numpy as np

from nilearn._base import NilearnBaseEstimator


class TestNilearnBaseEstimator(ABC):
    """Test class to test NilearnBaseEstimator methods."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.estimator = self.create_instance()

    @abstractmethod
    def create_instance(self) -> NilearnBaseEstimator:
        """Create estimator instance to test."""
        raise NotImplementedError()

    @pytest.fixture
    def fit_data(self) -> np.ndarray:
        """Return data to fit the estimator."""
        raise NotImplementedError()

    @pytest.fixture
    def fitted_estimator(self, fit_data) -> NilearnBaseEstimator:
        """Return a fitted estimator."""
        raise NotImplementedError()

    @pytest.fixture
    def doc(self) -> NumpyDocString:
        """Docstring of the estimator."""
        return NumpyDocString(self.estimator.__doc__)

    @pytest.fixture
    def documented_parameters(self, doc) -> dict[str, str]:
        """Parameters documented in the ``Parameters`` section of the class
        docstring.
        """
        return {param.name: param.type for param in doc["Parameters"]}

    @pytest.fixture
    def documented_attributes(self, doc) -> dict[str, str]:
        """Attributes documented in the ``Attributes`` section of the class
        docstring.
        """
        return {param.name: param.type for param in doc["Attributes"]}

    @pytest.fixture
    def parameters(self) -> list:
        """Public attributes of the estimator."""
        return [x for x in self.estimator.__dict__ if not x.startswith("_")]

    @pytest.fixture
    def undocumented_parameters(
        self, parameters, documented_parameters
    ) -> list:
        """The list of parameter names that appear in estimator attributes but
        not in ``Parameters`` section of the estimator docstring.
        """
        return list(set(parameters) - set(documented_parameters))

    @pytest.fixture
    def extra_parameters(self, parameters, documented_parameters) -> list:
        """The list of parameters that appear in ``Parameters`` section of the
        docstring of the estimator but not in the attributes of the estimator
        itself.
        """
        return list(set(documented_parameters) - set(parameters) - {"kwargs"})

    @pytest.fixture
    def fitted_attributes(self, fitted_estimator) -> list:
        """The list of attributes of the estimator after fit."""
        return [
            x
            for x in fitted_estimator.__dict__
            if x.endswith("_") and not x.startswith("_")
        ]

    @pytest.fixture
    def undocumented_attributes(
        self, fitted_attributes, documented_attributes
    ) -> list:
        """The list of attributes of the estimator after fit which do not
        appear in ``Attributes`` section of the estimator docstring.
        """
        return list(set(fitted_attributes) - set(documented_attributes))

    @pytest.fixture
    def extra_attributes(
        self, fitted_attributes, documented_attributes
    ) -> list:
        """The list of attributes that appear in ``Attributes`` section of the
        docstring but not in the attributes of the estimator after fit.
        """
        return list(set(documented_attributes) - set(fitted_attributes))

    def test_doc_sections_exist(self, doc) -> None:
        """Test if docstring of the estimator contains both ``Parameters`` and
        ``Attributes`` sections.
        """
        for section in ["Parameters", "Attributes"]:
            assert re.search(
                r"^" + section + "\n^-{10}", str(doc), re.MULTILINE
            ), (
                f"Estimator {self.estimator.__class__.__name__} "
                f"has no '{section} section."
            )

    def test_doc_no_duplicate_params(self, documented_parameters) -> None:
        """Test that there are no estimator parameters that are documented more
        than once in ``Parameters`` section of the estimator docstring.
        """
        assert len(documented_parameters) == len(set(documented_parameters))

    def test_doc_no_duplicate_attrs(self, documented_attributes) -> None:
        """Test that there are no estimator attributes that are documented more
        than once in ``Attributes`` section of the estimator docstring.
        """
        assert len(documented_attributes) == len(set(documented_attributes))

    def test_doc_no_undocumented_params(self, undocumented_parameters) -> None:
        """Test that there are no estimator attributes that are not documented
        in ``Parameters`` section of the estimator docstring.
        """
        assert not undocumented_parameters

    def test_doc_no_extra_params(self, extra_parameters) -> None:
        """Test that there are no parameters that appear in ``Parameters``
        section of the docstring of the estimator but not in the attributes of
        the estimator itself.
        """
        assert not extra_parameters

    def test_doc_param_order(self, documented_parameters) -> None:
        """Test the order of parameters in ``Parameters`` section of the
        estimator to follow the order of ``estimator.__init__ method
        parameters.
        """
        tmp = dict(**inspect.signature(self.estimator.__init__).parameters)

        assert [str(x) for x in documented_parameters] == [str(x) for x in tmp]

    def test_doc_attr_order(self, documented_attributes) -> None:
        """Test the order of the attributes documented in ``Attributes``
        section of the docstring to be in alphabetical order.

        Nice to have, but not always possible as sometimes doc string are
        composed from nilearn._utils.docs.
        """
        assert list(documented_attributes) != sorted(documented_attributes)

    def test_doc_no_undocumented_attributes_after_fit(
        self, undocumented_attributes
    ) -> None:
        """
        Test that all the attributes of the fitted estimator are documented in
        ``Attributes`` section of the estimator docstring.
        """
        assert not undocumented_attributes

    def test_doc_no_extra_attributes_after_fit(self, extra_attributes) -> None:
        """Test that all attributes in ``Attributes`` section of the docstring
        appear in estimator attributes after the estimator is fitted.
        """
        assert not extra_attributes

    def test_check_doc_link(self) -> None:
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

    def test_sklearn_tags(self) -> None:
        """Test the estimator has expected tags and corresponding values."""
        tags = self.estimator.__sklearn_tags__().input_tags
        assert hasattr(tags, "niimg_like")
        assert tags.niimg_like is False

    def test_verbose(self) -> None:
        """Test if the estimator's verbose attribute and the default value in
        ``__init__`` is 0.
        """
        assert self.estimator.verbose == 0

        signature = dict(
            **inspect.signature(self.estimator.__init__).parameters
        )
        assert signature["verbose"].default == 0
