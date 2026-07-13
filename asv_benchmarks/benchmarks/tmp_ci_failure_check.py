"""TEMPORARY benchmark used only to verify that the CI workflow's new
"Fail if any benchmark reported as failed" step actually fails the
workflow. Delete this file (and the matching "-b" debug filter in
.github/workflows/benchmark.yml) once that has been confirmed.
"""


class BenchMarkIntentionalCIFailure:
    """Always fails, on purpose, to check the CI failure-detection step."""

    def time_intentional_ci_failure(self):
        """Raise unconditionally."""
        raise RuntimeError("intentional failure to verify CI catches it")
