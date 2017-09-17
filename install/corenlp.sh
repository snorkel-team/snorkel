#!/usr/bin/env bash
set -exu

PARSER="stanford-corenlp-full-2015-12-09"

[ -e parser/corenlp.sh ] && exit

if ! [ -e  ~/download/${PARSER}.zip ] 
then
  url=http://nlp.stanford.edu/software/${PARSER}.zip
  mkdir -p ~/download
  curl --retry 10 -RL $url -o ~/download/${PARSER}.zip
fi

unzip ~/download/${PARSER}.zip
rm -rf parser
mv ${PARSER} parser
