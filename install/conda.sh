#!/usr/bin/env bash
set -ex

[ -e $HOME/miniconda/envs/py2snorkel ] && exit

curl --retry 10 -RL https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -o ~/download/miniconda.sh
apt-get update -qy && apt-get install -qy bzip2
bash ~/download/miniconda.sh -b -f -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda
conda create --yes -n py2snorkel
