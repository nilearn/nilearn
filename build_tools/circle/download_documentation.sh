#!/bin/bash

set -x -e

wget $GITHUB_ARTIFACT_URL
mkdir -p doc/_build/html
unzip doc*.zip -d doc/_build/html
