#!/bin/bash -exf

 GITHUB_HEAD_SHA=$(git log --no-merges -1 --pretty=format:%H)
if [ "$GITHUB_REF_NAME" == "main" ] || [[ $(cat gitlog.txt) == *"[circle full]"* ]]; then
    echo "Doing a full build";
    echo html-strict > build.txt;
else
    FILENAMES=$(git diff --name-only $(git merge-base $GITHUB_HEAD_SHA upstream/main) $GITHUB_HEAD_SHA);
    echo FILENAMES="$FILENAMES";
    for FILENAME in $FILENAMES; do
        if [[ `expr match $FILENAME "\(examples\)/.*plot_.*\.py"` ]]; then
            echo "Checking example $FILENAME ...";
            PATTERN=`basename $FILENAME`"\\|"$PATTERN;
        fi;
    done;
    echo PATTERN="$PATTERN";
    if [[ $PATTERN ]]; then
        # Remove trailing \| introduced by the for loop above
        PATTERN="\(${PATTERN::-2}\)";
        echo html-modified-examples-only > build.txt;
    else
        echo html-noplot > build.txt;
    fi;
fi;
echo "$PATTERN" > pattern.txt;
