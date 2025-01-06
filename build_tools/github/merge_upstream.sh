#!/bin/bash

# Add upstream as remote and merge main branch into the PR branch
# Generate a gitlog.txt file that stores the last commit message.

set -x -e

if ! git remote -v | grep upstream; then
    git remote add upstream https://github.com/nilearn/nilearn.git
fi
git fetch upstream

git log -1 --pretty=%B | tee gitlog.txt
echo "gitlog.txt = $(cat gitlog.txt)"

if [ -z ${CI+x} ]; then
    echo "Running locally";
    GITHUB_REF_NAME=$(git rev-parse --abbrev-ref HEAD)
    GITHUB_REF_TYPE="branch"
else
    echo "Running in CI";
fi

echo "$GITHUB_REF_NAME" | tee merge.txt
if [ "$GITHUB_REF_NAME" != "main" ] && [ "$GITHUB_REF_TYPE" == "branch" ]; then
    echo "Merging $(cat merge.txt)";
    git pull --ff-only upstream "refs/pull/$(cat merge.txt)";
fi
