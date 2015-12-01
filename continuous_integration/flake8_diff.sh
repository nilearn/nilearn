#!/bin/sh

set -e

# Travis does the git clone with a limited depth (50 at the time of
# writing). This may not be enough to find the common ancestor with
# $REMOTE/master so we unshallow the git checkout
git fetch --unshallow || echo "Unshallowing the git checkout failed"

# Tackle both common cases of origin and upstream as remote
# Note: upstream has priority if it exists
git remote -v
git remote | grep upstream && REMOTE=upstream || REMOTE=origin
git fetch -v $REMOTE master:remote_master

# Find common ancestor between HEAD and remote_master
COMMIT=$(git merge-base @ remote_master) || \
    echo "No common ancestor found for $(git show @ -q) and $(git show remote_master -q)"

if [ -z "$COMMIT" ]; then
    # clean-up created branch
    git branch -D remote_master
    exit 1
fi

echo Common ancestor is:
git show $COMMIT --stat

echo Running flake8 on the diff in the range\
     "$(git rev-parse --short $COMMIT)..$(git rev-parse --short @)" \
     "($(git rev-list $COMMIT.. | wc -l) commit(s)):"
git diff $COMMIT | flake8 --diff && echo -e "No problem detected by flake8\n"

# clean-up created branch
git branch -D remote_master
