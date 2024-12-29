#!/bin/bash -exf

#
# - Clone the doc repo in doc_build.
# - Deletes dev folder
# - Replaces it with content from doc/_build/html
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

echo "Deploying ${DEPLOY_TYPE} docs."
rm -fr "${CLONED_DOC:?}/${DEPLOY_TYPE}"
cp -a ./doc/_build/html "${CLONED_DOC}/${DEPLOY_TYPE}"
git -C ${CLONED_DOC} add -A

HEAD_COMMIT_MESSAGE=$(git log -1 --format=%s)
git -C ${CLONED_DOC} commit -m "${DEPLOY_TYPE} docs https://github.com/nilearn/nilearn/commit/${COMMIT_SHA} : ${HEAD_COMMIT_MESSAGE}"

git -C ${CLONED_DOC} push origin main

if [ "${DEPLOY_TYPE}" == "stable" ]; then
    VERSIONTAG=$(git describe --tags --abbrev=0)
    git -C ${CLONED_DOC} tag "${VERSIONTAG}" && \
	git -C ${CLONED_DOC} push --tags
fi
