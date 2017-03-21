#!/usr/bin/env bash

usage() {
	echo "usage: $0 [-h] PDF_FILE"
	echo "Script to extract coordinates at the word level from a PDF file"
	echo "Creates 2 files:"
	echo "	.ids_words.txt file with ids and corresponding word (id=page number + int)"
	echo "	.ids_coordinates.txt with ids and word coordinates on a page (top,left,bottom,right)" 
	exit 0
}	

if [ "$#" -ne 1 ]
then
	usage
elif [ $1 == "-h" ]
then
	usage	
fi 

INPUT_PDF=$1
NB_PAGES=$(pdfinfo $INPUT_PDF | grep Pages  | sed 's/[^0-9]*//')  
DIRNAME=$(dirname $INPUT_PDF)
FILENAME=$(basename $INPUT_PDF .pdf)
IDS_WORDS=$DIRNAME/$FILENAME.ids_words
IDS_COORDINATES=$DIRNAME/$FILENAME.ids_coordinates

for i in $(seq 1 $NB_PAGES);
do
	pdftotext -f $i -l $i -bbox-layout $INPUT_PDF $DIRNAME/$FILENAME.$i.html
	python parseHTMLoutput.py $DIRNAME/$FILENAME.$i.html $IDS_WORDS.$i.txt $IDS_COORDINATES.$i.txt $i
	cat $IDS_WORDS.$i.txt >> $IDS_WORDS.txt
	cat $IDS_COORDINATES.$i.txt >> $IDS_COORDINATES.txt
	rm $DIRNAME/$FILENAME.$i.html
	rm $IDS_WORDS.$i.txt
	rm $IDS_COORDINATES.$i.txt
done
