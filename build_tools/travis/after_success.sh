#!/bin/sh

set -e

# Ignore codecov failures because we don't want travis to report a failure
# in the github UI just because the coverage report failed to be published.
# codecov needs to be run from the git checkout
# so we need to copy the coverage results from TEST_RUN_FOLDER
if [[ "$SKIP_TESTS" != "true" && "$COVERAGE" == "true" ]]; then
    cp "$TEST_RUN_FOLDER/.coverage" .
    codecov || echo "Codecov upload failed"
fi
