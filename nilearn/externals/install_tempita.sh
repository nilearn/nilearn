#!/bin/sh
# Script to do a local install of tempita
set +x
export LC_ALL=C
INSTALL_FOLDER=tmp/tempita_install
rm -rf tempita $INSTALL_FOLDER
if [ -z "$1" ]
then
        TEMPITA=tempita
else
        TEMPITA=$1
fi

pip install --no-cache $TEMPITA --target $INSTALL_FOLDER
cp -r $INSTALL_FOLDER/tempita tempita
rm -rf $INSTALL_FOLDER

# Needed to rewrite the doctests
# Note: BSD sed -i needs an argument unders OSX
# so first renaming to .bak and then deleting backup files
find tempita -name "*.py" | xargs sed -i.bak "s/from tempita/from nilearn.externals.tempita/"
find tempita -name "*.bak" | xargs rm
