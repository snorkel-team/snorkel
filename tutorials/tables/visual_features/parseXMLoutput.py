import argparse
import codecs 
from bs4 import BeautifulSoup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
		    description="Script to extract words coordinates in an xml file")
    parser.add_argument("input_xml", help="xml file to parse")
    parser.add_argument("ouput_ids", help="text file containing unique ids and corresponding word")
    parser.add_argument("ouput_coord", help="text file containing unique ids and coordinates (x,y,height,width)")
    parser.add_argument("pdf_page_number")
    parser.add_argument("--delimiter", help="delimiter in output files", default="\t") 
    args = parser.parse_args()
    xml_in = open(args.input_xml, 'r').read()
    #xml_in = codecs.open(args.input_xml, encoding='utf-8').read()  #TODO fix this 
    ids_words = open(args.ouput_ids, 'w')
    ids_coordinates = open(args.ouput_coord, 'w')
    page_nb = args.pdf_page_number
    delimiter = args.delimiter
    soup = BeautifulSoup(xml_in, "xml")
    texts = soup.find_all("text")
    i = 0
    for text in texts:
        h = text.get('height')
	x = text.get('left')
	w = text.get('width')
	y = text.get('top')
	content = text.getText().split(' ')
	if len(content) == 1:
		# If only one word in the tag
		try: 
			ids_words.write(page_nb+str(i) + delimiter + content[0] + '\n')
			ids_coordinates.write(delimiter.join([page_nb+ str(i), str(x), str(y),str(h),str(w)]) + '\n')
			i +=1
		except UnicodeEncodeError:
			pass
	else:
		# Multiple words in one box 
		char_width = float(w)/len(text.getText())
		for word in content:
			try:
				word_width = int(char_width*len(word))
				ids_words.write(page_nb+str(i) + delimiter + word + '\n')
				ids_coordinates.write(delimiter.join([page_nb+str(i), str(x), str(y),str(h),str(word_width)])+ '\n')
				i += 1
				x = str(float(x) + word_width + 1) # add 1 for white space
			except UnicodeEncodeError:
				pass
    ids_words.close()
    ids_coordinates.close()

