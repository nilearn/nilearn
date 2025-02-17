#!/bin/bash -exf

# - Clone the doc repo in doc_build.
#
# For deploy of doc build on PR (dev)
# - Deletes dev folder
# For deploy of doc build on release (stable)
# - Deletes stable folder
#
# - Replaces it with content from doc/_build/html
# For deploy of doc build on release (stable)
# - Also copies the doc in a folder X.Y.Z with X.Y.Z corresponding to the doc version
#
# - Commits changes and push
#
# USAGE
#
# From the root of the repo
#
#  # Set env variables
#  HEAD_COMMIT_MESSAGE='foo'
#  COMMIT_SHA='xxxx'
#
#  # must be dev or stable
#  DEPLOY_TYPE="dev"
#
#  bash ./build_tools/github/deploy_doc.sh
#

set -x -e

CLONED_DOC='doc/_build/nilearn.github.io'

if [ ! -d ${CLONED_DOC} ]; then
    echo "Cloning nilearn/nilearn.github.io: this may take a while..."
    # --depth 1 is a speed optimization
    # since we don't need the history prior to the last commit
    git clone git@github.com:nilearn/nilearn.github.io.git ${CLONED_DOC} --depth=1
fi

git -C ${CLONED_DOC} checkout main

git -C ${CLONED_DOC} remote -v

git -C ${CLONED_DOC} fetch origin
git -C ${CLONED_DOC} reset --hard origin/main
git -C ${CLONED_DOC} clean -xdf

echo "Copying ${DEPLOY_TYPE} docs."

rm -fr "${CLONED_DOC:?}/${DEPLOY_TYPE}"

cp -a ./doc/_build/html "${CLONED_DOC}/${DEPLOY_TYPE}"

if [ "${DEPLOY_TYPE}" == "stable" ]; then
    VERSIONTAG=$(git describe --tags --abbrev=0)
    cp -a ./doc/_build/html "${CLONED_DOC}/${VERSIONTAG}"
fi

echo "Deploying ${DEPLOY_TYPE} docs."

git -C ${CLONED_DOC} add -A

if [ "${DEPLOY_TYPE}" == "stable" ]; then
    MSG="${VERSIONTAG}"
else
    HEAD_COMMIT_MESSAGE=$(git log -1 --format=%s)
    MSG="https://github.com/nilearn/nilearn/commit/${COMMIT_SHA} : ${HEAD_COMMIT_MESSAGE}"
fi
git -C ${CLONED_DOC} commit -m "${DEPLOY_TYPE} docs: ${MSG}"

git -C ${CLONED_DOC} push origin main

if [ "${DEPLOY_TYPE}" == "stable" ]; then
    git -C ${CLONED_DOC} tag "${VERSIONTAG}" && \
	git -C ${CLONED_DOC} push --tags
fi
