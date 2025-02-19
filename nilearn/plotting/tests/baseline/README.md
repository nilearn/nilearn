# Baseline figure for plotting tests

This folder contain the images used as 'baseline' for several plotting tests.

Sometimes, the output of a plotting function may unintentionally change
as a side effect of changing another function or piece of code
that it depends on.

For several tests we ensure that the outputs are not accidentally changed.

Failures are expected at times when the output is changed intentionally
(e.g. fixing a bug or adding features) for a particular function.

In such cases, the output needs to be manually/visually checked
as part of the PR review process and then a new baseline set for comparison.

Set a new baseline by running the following
with the oldest supported python:

```
pip install tox
tox run -e pytest_mpl_generate
```
