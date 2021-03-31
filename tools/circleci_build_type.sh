#!/bin/bash -ef

if [ "CIRCLE_BRANCH" == "master" ] || [[ $(cat gitlog.txt) == *"[circle full]"* ]]; then
    echo "Doing a full build";
    echo html-strict > build.txt;
else
    echo "Doing a partial build";
    FILENAMES=$(git diff --name-only $(git merge-base $CIRCLE_BRANCH upstream/master) $CIRCLE_BRANCH);
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
        echo html-pattern > build.txt;
    else
        echo html-noplot > build.txt;
    fi;
fi;
echo "$PATTERN" > pattern.txt;

