#!/bin/bash -exf

# Update version of niivue vendored with Nilearn
# according to the version found in npm-requirements.txt
#
# requires npm

set -x -e

npm install "$(cat npm-requirements.txt)"

if [ -f  "nilearn/plotting/data/js/niivue.umd.js" ]; then
    rm nilearn/plotting/data/js/niivue.umd.js
fi

if [ -f  "node_modules/@niivue/niivue/node_modules/fflate/umd/index.js" ]; then
    source_file="node_modules/@niivue/niivue/node_modules/fflate/umd/index.js"

elif [ -f  "node_modules/@niivue/niivue/dist/niivue.umd.js" ]; then
    source_file="node_modules/@niivue/niivue/dist/niivue.umd.js"
fi

cp "$source_file" nilearn/plotting/data/js/niivue.umd.js

NEW_VERSION=$(grep "@niivue/niivue" npm-requirements.txt | awk -F'@' '{print $3}')

sed -i "s/NIIVUE_VERSION=\"[0-9]\+\.[0-9]\+\.[0-9]\+\"/NIIVUE_VERSION=\"$NEW_VERSION\"/" nilearn/plotting/js_plotting_utils.py
