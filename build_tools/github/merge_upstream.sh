#!/bin/bash

# Add upstream as remote and merge main branch into the PR branch

set -x -e

if ! git remote -v | grep upstream; then
    git remote add upstream https://github.com/nilearn/nilearn.git
fi
git fetch upstream

git log -1 --pretty=%B | tee gitlog.txt
echo "gitlog.txt = $(cat gitlog.txt)"

echo "$GITHUB_REF_NAME" | tee merge.txt
if [ "$GITHUB_REF_NAME" != "main" ]; then
    echo "Merging $(cat merge.txt)";
    git pull --ff-only upstream "refs/pull/$(cat merge.txt)";
fi
