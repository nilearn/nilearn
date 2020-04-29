#!/bin/sh
# Script to do a local install of brainsprite
set +x
export LC_ALL=C
INSTALL_FOLDER=tmp/brainsprite_install
rm -rf brainsprite $INSTALL_FOLDER
if [ -z "$1" ]
then
        BRAINSPRITE=brainsprite
else
        BRAINSPRITE=$1
fi

pip install --no-cache $BRAINSPRITE --target $INSTALL_FOLDER
cp -r $INSTALL_FOLDER/brainsprite brainsprite
rm -rf $INSTALL_FOLDER

# Needed to rewrite the doctests
# Note: BSD sed -i needs an argument unders OSX
# so first renaming to .bak and then deleting backup files
find brainsprite -name "*.py" | xargs sed -i.bak "s/from brainsprite/from nilearn.externals.brainsprite/"
find brainsprite -name "*.bak" | xargs rm
