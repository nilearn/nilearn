#!/bin/bash -x

if [[ -n "$FLAKE8" ]]; then
    echo "Running flake8 diff script...";
    source build_tools/flake8_diff.sh
fi

python -m pytest --pyargs nilearn --cov=nilearn

if [[ $TEST_DOC == true ]]; then
    echo "Running make test-doc...";
    make test-doc
fi
