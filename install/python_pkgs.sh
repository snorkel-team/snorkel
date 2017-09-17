#!/usr/bin/env bash
set -ex

export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/miniconda_env.sh

# Packages required by Snorkel
conda install --yes numba
if [ "$TRAVIS" = "true" ]
then
  # Install binary distribution of scientific python modules
  conda install --yes numpy scipy matplotlib pip
fi

pip install -r python-package-requirement.txt

# Download spaCy English model
python -m spacy download en

jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Use runipy to run Jupyter/IPython notebooks from command-line
pip install runipy

conda list
