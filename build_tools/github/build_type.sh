#!/bin/bash -exf

GITLOG=$(cat gitlog.txt)
if [ "$GITHUB_REF_NAME" == "main" ] || [[ $GITLOG == *"[full doc]"* ]]; then
    echo "Doing a full build";
    echo html-strict > build.txt;
else
    if [[ $GITLOG == *"[example]"* ]]; then
        echo "Building selected example";
        COMMIT_MESSAGE=${GITLOG#*] };
        FILENAMES="examples/*/$COMMIT_MESSAGE";
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
    else
        echo html-noplot > build.txt;
    fi;
fi;
echo "$PATTERN" > pattern.txt;
