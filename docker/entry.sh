#!/usr/bin/env bash
set -ex

source set_env.sh
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
