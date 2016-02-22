#!/bin/sh

set -e

# Ignore coveralls failures as the coveralls server is not very reliable
# but we don't want travis to report a failure in the github UI just
# because the coverage report failed to be published.
# coveralls need to be run from the git checkout
# so we need to copy the coverage results from TEST_RUN_FOLDER
if [[ "$SKIP_TESTS" != "true" && "$COVERAGE" == "true" ]]; then
    cp "$TEST_RUN_FOLDER/.coverage" .
    coveralls || echo "Coveralls upload failed"
fi
