#!/bin/bash -exf

if [ "$GITHUB_REF_NAME" == "main" ] || [[ $(cat gitlog.txt) == *"[full doc]"* ]]; then
    echo "Doing a full build";
    echo html-strict > build.txt;
else
    if [[ $(cat gitlog.txt) == *"[examples]"* ]]; then
        COMMIT_MESSAGE=${$(cat gitlog.txt)#*]};
        if [[ $COMMIT_MESSAGE == "^[0-9][0-9]" ]]; then
            DIRECTORY="../examples/${COMMIT_MESSAGE}";
        else
            FILENAMES="examples/*/${COMMIT_MESSAGE}";
        fi;
    else
        FILENAMES=$(git diff --name-only $(git merge-base $COMMIT_SHA upstream/main) $COMMIT_SHA);
    fi;
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
    elif [[ $DIRECTORY ]]; then
        PATTERN="$DIRECTORY"
        echo html-modified-directory > build.txt;
    else
        echo html-noplot > build.txt;
    fi;
fi;
echo "$PATTERN" > pattern.txt;
