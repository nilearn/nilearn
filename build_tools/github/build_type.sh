#!/bin/bash -ef

if [ "$GITHUB_REF_NAME" == "main" ] || [[ $(cat gitlog.txt) == *"[circle full]"* ]]; then
    echo "Doing a full build";
    echo html-strict > build.txt;
else
    echo "Doing a partial build";
    FILENAMES=$(git diff --name-only $(git merge-base $GITHUB_REF_NAME upstream/main) $GITHUB_REF_NAME);
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
