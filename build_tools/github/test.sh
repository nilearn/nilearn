#!/bin/bash -x

if [[ -n "$FLAKE8" ]]; then
    echo "Running flake8 diff script...";
    source build_tools/flake8_diff.sh
fi

python -m pytest --pyargs nilearn --cov=nilearn

make test-doc
