#!/bin/sh

set -e

if [[ -n "$FLAKE8_VERSION" ]]; then
    source continuous_integration/flake8_diff.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    python continuous_integration/show-python-packages-versions.py
    # Copy setup.cfg to TEST_RUN_FOLDER where we are going to run the tests from
    # Mainly for nose config settings
    cp setup.cfg "$TEST_RUN_FOLDER"
    # We want to back out of the current working directory to make
    # sure we are using nilearn installed in site-packages rather
    # than the one from the current working directory
    # Parentheses (run in a subshell) are used to leave
    # the current directory unchanged
    (cd "$TEST_RUN_FOLDER" && make -f $OLDPWD/Makefile test-code)
    test "$MATPLOTLIB_VERSION" == "" || make test-doc
fi
