set -eu
PARSER="stanford-corenlp-full-2016-10-31"
rm -rf parser
url=http://nlp.stanford.edu/software/${PARSER}.zip
if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N -nc $url
fi
unzip ${PARSER}.zip
mv ${PARSER} parser
