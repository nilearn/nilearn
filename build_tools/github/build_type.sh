#!/bin/bash -exf

set -x -e

GITLOG=$(cat gitlog.txt)

# pattern.txt lists the examples to build
# must be generated even if empty
# as its content is passed to sphinx_gallery_conf.filename_pattern
# via a PATTERN env variable
touch pattern.txt;

# do a full build on main, release or if requested in the commit message
# exit early
if [ "$GITHUB_REF_NAME" == "main" ] || [ "$GITHUB_REF_TYPE" != "tag" ] || [[ $GITLOG == *"[full doc]"* ]]; then
    echo "Doing a full build";
    echo html-strict > build.txt;
    exit 1
fi

# check if the build of some examples was requested in the commit message
# like:
# git commit -m '[example] plot_*atlas*.py'
EXAMPLE=""
if [[ $GITLOG == *"[example]"* ]]; then
    echo "Building selected example";
    COMMIT_MESSAGE=${GITLOG#*] };
    EXAMPLE="examples/*/$COMMIT_MESSAGE";
fi;

if [ -z ${CI+x} ]; then
    echo "Running locally";
    COMMIT_SHA=$(git log --format=format:%H -n 1)
fi

# generate examples.txt that will list
# - all the files modified in this PR
# - and the examples listed in the commit message
git diff --name-only "$(git merge-base $COMMIT_SHA upstream/main)" "$COMMIT_SHA" | tee examples.txt;
echo "$EXAMPLE" >> examples.txt

# Filter the list of files to only keep files
# that are of interest for sphinx_gallery
for FILENAME in $(cat examples.txt); do
    if [[ $(expr match "$FILENAME" "\(examples\)/.*plot_.*\.py") ]]; then
        echo "Checking example $FILENAME ...";
        PATTERN=$(basename "$FILENAME")"\\|"$PATTERN;
    fi;
done;
echo PATTERN="$PATTERN";

if [[ $PATTERN ]]; then
    # Remove trailing \| introduced by the for loop above
    PATTERN="\(${PATTERN::-2}\)";
    echo html-modified-examples-only > build.txt;
else
    echo ci-html-noplot > build.txt;
fi;

echo "$PATTERN" > pattern.txt;
