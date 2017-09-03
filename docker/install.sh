#!/usr/bin/env bash
set -ex

function install_python_pkgs {
  # Packages required by Snorkel
  # apt-get update
  # apt-get install -y g++ # for compiling SpaCy
  pip install --no-cache-dir -r python-package-requirement.txt
  jupyter nbextension enable --py widgetsnbextension --sys-prefix
  pip install --no-cache-dir runipy
}

function install_corenlp {
  test -e parser/corenlp.sh && return 0
  install_jdk
  apt-get install -y unzip
  ./install-parser.sh
}

# JDK for running CoreNLP
function install_jdk {
  apt-get update
  apt-get install -y software-properties-common
  add-apt-repository "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main"
  apt-get update
  echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections
  apt-get install -y oracle-java8-installer
  java -version
}

function cleanup {
  # apt-get purge -y g++ unzip
  apt-get autoremove -y
  rm -rf /var/lib/apt/lists/*
  rm -rf /var/lib/dpkg/info/*
  rm -rf /tmp/*
  rm -rf /root/.cache/*
}
