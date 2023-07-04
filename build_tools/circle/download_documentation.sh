#!/bin/bash

set -e
set -x

wget $GITHUB_ARTIFACT_URL
mkdir -p doc/_build/html
unzip doc*.zip -d doc/_build/html
