PARSER="stanford-corenlp-full-2015-12-09"
rm -rf parser
wget http://nlp.stanford.edu/software/${PARSER}.zip
unzip ${PARSER}.zip
mv ${PARSER} parser
