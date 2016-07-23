#!/usr/bin/env bash
# test script for Snorkel
set -euo pipefail
Here=$(dirname "$0")

# run all .bats files under test/ unless arguments specify which ones to run
[[ $# -gt 0 ]] || { cd "$Here"; set -- */*.bats; }

# ensure environment
set -x
cd "$Here"

# use bats to run everything
test/bats/bin/bats "$@"
