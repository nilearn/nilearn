#!/bin/bash
#
# This script is used to check
# if there are any unreplaced arguments from the docstrings
# in html version of the docs.
#
#   Parameters
#   ----------
#   %(img)s    < ----- argument to replace by strings defined in nilearn/_utils/docs.py
#
# If the output of this script is empty, then there are no unreplaced arguments.
# If not, then some function is likely missing an @fill_doc decorator.
#
# See nilearn/_utils/docs.py.

set -e
set -x

mkdir -p doc/tmp
grep --include "*.html" -rn "doc/_build/html/" -e "%(" > doc/tmp/doc_check.txt || true
