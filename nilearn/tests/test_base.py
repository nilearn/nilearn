import inspect
import re
from abc import ABC, abstractmethod

import numpy as np
import pytest
from joblib import Memory
from numpydoc.docscrape import NumpyDocString
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._base import NilearnBaseEstimator


def sklearn_estimator_checks(cls):

    def test_run_sklearn_checks(self, estimator, check):
        check(estimator)

    decorated_test = parametrize_with_checks(
        estimators=cls.estimator_list,
        expected_failed_checks=lambda est: cls.expected_failed_check_list,
    )(test_run_sklearn_checks)

    cls.test_sklearn_compatible_estimator = decorated_test

    return cls


class TestBaseEstimator(ABC):
    """Test class to run scikit-learn checks on the estimator."""

    estimator_list: list[NilearnBaseEstimator] = []
    expected_failed_check_list: dict[str, str] = {}


class TestNilearnBaseEstimator(TestBaseEstimator):
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


class TestCacheMixin:
    @pytest.fixture
    def tmp_cache_dir(self, tmp_path) -> str:
        return tmp_path / "cache_dir"

    @pytest.fixture
    def cache_dir_str(self) -> str:
        return "~/nilearn_data/test_cache"

    @pytest.fixture
    def memory(self, tmp_cache_dir):
        return Memory(location=tmp_cache_dir)

    @pytest.fixture
    def estimator(self):
        raise NotImplementedError()

    @pytest.fixture
    def estimator_with_memory(self, memory):
        raise NotImplementedError()

    def test_memory_at_init(self, estimator) -> None:
        assert estimator.memory is None
        assert estimator.memory_level == 0

    def test_fit_cache_memory_none(self, estimator) -> None:
        estimator._fit_cache()
        assert estimator.memory is None
        assert estimator.memory_level == 0

        assert isinstance(estimator.memory_, Memory)
        assert estimator.memory_.location is None

    def test_fit_cache_memory_not_none(
        self, estimator_with_memory, tmp_cache_dir
    ) -> None:
        estimator_with_memory._fit_cache()
        assert isinstance(estimator_with_memory.memory, Memory)
        assert estimator_with_memory.memory.location == tmp_cache_dir
        assert estimator_with_memory.memory_level == 0

        assert isinstance(estimator_with_memory.memory_, Memory)
        assert estimator_with_memory.memory_.location == tmp_cache_dir
        assert estimator_with_memory.memory == estimator_with_memory.memory_

    def test_cache_memory_not_none_level_0(
        self, estimator_with_memory, tmp_cache_dir
    ) -> None:
        assert estimator_with_memory.memory_level == 0
        with pytest.warns(
            UserWarning,
            match="memory_level is currently set to 0 but a Memory object",
        ):
            cached_func = estimator_with_memory._cache(None)
        estimator_with_memory.memory_level = 1
        assert cached_func is not None

    def test_cache_memory_not_none_level_1(
        self, recwarn, estimator_with_memory, tmp_cache_dir
    ) -> None:
        estimator_with_memory.memory_level = 1
        cached_func = estimator_with_memory._cache(None)
        estimator_with_memory.memory_level = 1
        assert cached_func is not None
        assert len(recwarn) == 0
