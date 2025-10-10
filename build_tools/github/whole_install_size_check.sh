#!/usr/bin/env bash
set -euo pipefail

# script to check size of the whole install of nilearn
# and all its oldest dependencies and transitive dependencies
# does not get too large.

THRESHOLD=$((560 * 1024 * 1024))   # adjust as needed

rm -rf dist build *.egg-info

TMPENV=$(mktemp -d)
python -m venv "$TMPENV/venv"
source "$TMPENV/venv/bin/activate"

echo "virtual env in ${TMPENV}/venv"

pip install --upgrade pip
pip install .[min_plotting]

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
TOTAL=$(du -sb "$SITE" 2>/dev/null | cut -f1 || du -sk "$SITE" | awk '{print $1*1024}')
TOTAL_MB=$(awk "BEGIN {print $TOTAL/1024/1024}")

echo "Full installed environment size: ${TOTAL_MB} MB (threshold: $(($THRESHOLD/1024/1024)) MB)"

deactivate

if [ "$TOTAL" -gt "$THRESHOLD" ]; then
    echo "❌ whole install size check failed"
    exit 1
fi

# only remove the virtual env if check passes
# in case we want to inspect things if it fails
rm -rf "$TMPENV"

echo "✅ whole install size check passed"
exit 0
