#!/bin/bash

set -e -x -u -o pipefail

mkdir -p tmp
grep --include "*.html" -rn "_build/html/" -e "%(" > tmp/doc_check.txt
