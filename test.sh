#!/usr/bin/env bash
# test script for Snorkel
set -euo pipefail
set -x

# Run notebooks end-to-end
# NOTE: Currently some tests (LSTMTests) rely on data produced by these notebooks...
TESTING=true runipy tutorial/CDR_Tutorial.ipynb
runipy examples/GeneTaggerExample_Extraction.ipynb  
runipy examples/GeneTaggerExample_Learning.ipynb

# Run test modules
python test/ParserTests.py
python test/CandidateSpaceTests.py
python test/MatcherTests.py
python test/InferenceTests.py
python test/LSTMTests.py

# TODO check outputs, upload results, etc.
# for more ideas, see: https://github.com/rossant/ipycache/issues/7
