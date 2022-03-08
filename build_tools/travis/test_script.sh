#!/bin/sh

set -e

TEST_CMD="pytest --pyargs"

if [[ $TRAVIS_CPU_ARCH == arm64 ]]; then
    TEST_CMD="$TEST_CMD -n $CPU_COUNT"
fi

$TEST_CMD nilearn
