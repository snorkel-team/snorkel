#!/usr/bin/env bash
set -ex

apt-get update -qy
apt-get install -qy curl wget g++ git python-tk unzip libgl1-mesa-glx \
  software-properties-common

add-apt-repository "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main"
apt-get update -qy
echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections
apt-get install -qy --allow-unauthenticated oracle-java8-installer
java -version
