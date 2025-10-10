#!/usr/bin/env bash

# script to check size of the nilearn wheel to avoid the package getting too large
set -x -e

THRESHOLD=$((11 * 1024 * 1024))   # 11 MB max

rm -rf dist build *.egg-info

python -m pip install --upgrade pip
pip install --prefer-binary build
python -m build --wheel

WHEEL=$(ls dist/nilearn-*.whl || true)
SIZE=$(stat -c%s "$WHEEL" 2>/dev/null || stat -f%z "$WHEEL")
SIZE_MB=$(awk "BEGIN {print $SIZE/1024/1024}")

echo "Wheel size: $SIZE_MB MB (threshold: $(($THRESHOLD/1024/1024)) MB)"

if [ "$SIZE" -gt "$THRESHOLD" ]; then
    echo "❌ wheel size check failed"
    exit 1
fi

echo "✅ wheel size check passed"
exit 0
