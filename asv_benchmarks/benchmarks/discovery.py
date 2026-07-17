"""Benchmarks for nilearn.utils.discovery."""


class BenchMarkAllEstimators:
    """Check discovery of all nilearn estimators."""

    def setup(self):
        """Set up for all benchmarks."""
        # Imported locally (nilearn.utils.discovery module was only added
        # in 0.13.0) so that benchmarking older nilearn versions only
        # affects this benchmark instead of making the whole module fail
        # to import for every benchmark in this file.
        # Raising NotImplementedError (instead of letting ImportError
        # propagate) makes asv report this as "skipped" rather than
        # "failed" for those older versions: see asv_runner's
        # BenchmarkBase.do_setup, which special-cases NotImplementedError.
        try:
            from nilearn.utils import all_estimators
        except ImportError as e:
            raise NotImplementedError from e

        self.all_estimators = all_estimators

    def time_discovery_all_estimators(self):
        """Check time."""
        self.all_estimators()

    def peakmem_discovery_all_estimators(self):
        """Check peak memory."""
        self.all_estimators()
