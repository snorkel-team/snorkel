#!/usr/bin/env bash
set -ex

# apt-get purge -y g++ unzip
apt-get autoremove -y
rm -rf /var/lib/apt/lists/*
rm -rf /var/lib/dpkg/info/*
rm -rf /tmp/*
rm -rf /root/.cache/*
rm -rf /root/download/*
