#!/bin/bash -exf

set -x -e

rm -fr dist

python -m pip install --upgrade pip
pip install --prefer-binary build
python -m build
ls -lrth dist

# Path to newly built wheel
NEW_WHEEL=$(ls dist/nilearn-*-py3-none-any.whl)

# Get size of new wheel in bytes
NEW_SIZE=$(stat -c%s "$NEW_WHEEL" || stat -f%z "$NEW_WHEEL")

echo "New wheel size: $NEW_SIZE bytes"

# Download latest wheel from PyPI
pip install --upgrade pip
pip download nilearn --only-binary=:all: --no-deps -d /tmp
OLD_WHEEL=$(ls /tmp/nilearn-*-py3-none-any.whl)

OLD_SIZE=$(stat -c%s "$OLD_WHEEL" || stat -f%z "$OLD_WHEEL")

echo "Latest PyPI wheel size: $OLD_SIZE bytes"

# Allow up to 10% increase
LIMIT=$(( OLD_SIZE * 105 / 100 ))

if [ "$NEW_SIZE" -gt "$LIMIT" ]; then
    echo "❌ Wheel size regression detected!"
    echo "New:  $NEW_SIZE bytes"
    echo "Old:  $OLD_SIZE bytes"
    exit 1
else
    echo "✅ Wheel size check passed"
fi

# Compare installed sizes
echo "Checking installed package size..."
TMPENV=$(mktemp -d)

# Install new wheel in a fresh venv
python -m venv $TMPENV/newenv
source $TMPENV/newenv/bin/activate
pip install "$NEW_WHEEL"
NEW_INSTALLED=$(du -sb $(python -c "import nilearn, os; print(os.path.dirname(nilearn.__file__))") | cut -f1)
deactivate

# Install PyPI wheel in another fresh venv
python -m venv $TMPENV/oldenv
source $TMPENV/oldenv/bin/activate
pip install nilearn
OLD_INSTALLED=$(du -sb $(python -c "import nilearn, os; print(os.path.dirname(nilearn.__file__))") | cut -f1)
deactivate

echo "New installed size: $NEW_INSTALLED bytes"
echo "Old installed size: $OLD_INSTALLED bytes"

LIMIT=$(( OLD_INSTALLED * 105 / 100 ))
if [ "$NEW_INSTALLED" -gt "$LIMIT" ]; then
    echo "❌ Installed size regression detected!"
    echo "New: $NEW_INSTALLED bytes vs Old: $OLD_INSTALLED bytes"
    exit 1
else
    echo "✅ Installed size check passed"
fi
