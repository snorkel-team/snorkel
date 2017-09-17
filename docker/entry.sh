#!/usr/bin/env bash
set -ex

source /snorkel/docker/set_env.sh
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
