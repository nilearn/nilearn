#!bin/bash
source activate testenv

# pipefail is necessary to propagate exit codes
set -o pipefail && cd doc && make html-strict 2>&1 | tee ~/log.txt
