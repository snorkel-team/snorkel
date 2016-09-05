#!/bin/sh
mkdir downloads
cd downloads
echo "Downloading CDR data..."
wget http://www.biocreative.org/media/store/files/2016/CDR_Data.zip
echo "Downloading CTD data..."
wget http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz
echo "Extracting data..."
unzip -qq CDR_Data.zip
gunzip -q CTD_chemicals_diseases.tsv.gz
echo "Copying data to data directory..."
cd ../
cp downloads/CTD_chemicals_diseases.tsv data/dicts/.
cp downloads/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml data/.
cp downloads/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml data/.
cp downloads/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml data/.
echo "Deleting downloads directory..."
rm -rf downloads
echo "Done!"

