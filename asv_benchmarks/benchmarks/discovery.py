"""Benchmarks for nilearn.utils.discovery."""


class BenchMarkAllEstimators:
    """Check discovery of all nilearn estimators."""

    def setup(self):
        """Set up for all benchmarks."""
        # Imported locally (nilearn.utils.discovery module was only added
        # in 0.13.0) so that benchmarking older nilearn versions only
        # fails this benchmark instead of making the whole module fail
        # to import for every benchmark in this file.
        from nilearn.utils import all_estimators

        self.all_estimators = all_estimators

    def time_discovery_all_estimators(self):
        """Check time."""
        self.all_estimators()

    def peakmem_discovery_all_estimators(self):
        """Check peak memory."""
        self.all_estimators()
