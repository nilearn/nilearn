#!/bin/bash
#
# This script is used to check
# if the warnings thrown in the examples.
#
# It aims to help minimize the number of warnings thrown.
#

set -e
set -x

mkdir -p doc/tmp
grep --include "*.html" -rn "doc/_build/html/auto_examples" -e "Warning: " > doc/tmp/all_warnings.txt || true
grep --include "*.html" -rn "doc/_build/html/auto_examples" -e "DeprecationWarning: " > doc/tmp/deprecation_warnings.txt || true
grep --include "*.html" -rn "doc/_build/html/auto_examples" -e "FutureWarning: " > doc/tmp/future_warnings.txt || true
grep --include "*.html" -rn "doc/_build/html/auto_examples" -e "UserWarning: " > doc/tmp/user_warnings.txt || true
grep --include "*.html" -rn "doc/_build/html/auto_examples" -e "RuntimeWarning: " > doc/tmp/runtime_warnings.txt || true
