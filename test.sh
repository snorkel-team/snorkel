#!/usr/bin/env bash
set -ex

source set_env.sh

# Run test modules
python test/learning/test_gen_learning.py
python test/learning/test_supervised.py
python test/learning/test_categorical.py
runipy test/learning/test_TF_notebook.ipynb
runipy test/learning/test_parallel_grid_search.ipynb

# Runs intro tutorial notebooks
cd tutorials
runipy intro/Intro_Tutorial_1.ipynb
runipy intro/Intro_Tutorial_2.ipynb
runipy intro/Intro_Tutorial_3.ipynb

# Run advanced notebooks
runipy advanced/Categorical_Classes.ipynb
runipy advanced/Structure_Learning.ipynb

# Run CDR tutorials
runipy cdr/CDR_Tutorial_1.ipynb
runipy cdr/CDR_Tutorial_2.ipynb
runipy cdr/CDR_Tutorial_3.ipynb

# TODO check outputs, upload results, etc.
# for more ideas, see: https://github.com/rossant/ipycache/issues/7
