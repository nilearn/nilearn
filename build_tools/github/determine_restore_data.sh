#!/bin/bash

# Check if gtihub cache can be used
#
# If the commit message contains "force download" then cache is ignored
# and all data will be downloaded.
#
# If the build type is a no plot in CI, then cache is not used.

set -e -x

commit_msg=$(git log -2 --format=oneline);
echo $commit_msg;

if [[ $commit_msg == *"[force download]"* ]]; then
    echo "All datasets will be downloaded as requested (only for full builds).";
    echo "false" | tee restore.txt;
else
    echo "Data cache will be used if available.";
    echo "true" | tee restore.txt;
fi
