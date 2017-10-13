#!/bin/sh
mkdir downloads
cd downloads
echo "Downloading CDR data..."
wget http://www.biocreative.org/media/store/files/2016/CDR_Data.zip
echo "Extracting data..."
unzip -qq CDR_Data.zip
echo "Copying data to data directory..."
cd ../
cp downloads/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml data/.
cp downloads/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml data/.
cp downloads/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml data/.
echo "Deleting downloads directory..."
rm -rf downloads
echo "Combining data files..."
{
	head -69462 data/CDR_TrainingSet.BioC.xml;
	tail -70514 data/CDR_DevelopmentSet.BioC.xml | head -70512;
	tail -71613 data/CDR_TestSet.BioC.xml;
} > data/CDR.BioC.xml
echo "Done!"

