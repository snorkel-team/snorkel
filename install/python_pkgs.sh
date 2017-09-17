#!/usr/bin/env bash
set -ex

export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/miniconda_env.sh

# Install binary distribution of scientific python modules
conda config --add channels conda-forge
conda install --yes \
  numba numpy scipy matplotlib \
  tensorflow pandas \
  spacy nltk jupyter

pip install -r python-package-requirement.txt

# Download spaCy English model
python -m spacy download en

jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Use runipy to run Jupyter/IPython notebooks from command-line
pip install runipy

conda list
