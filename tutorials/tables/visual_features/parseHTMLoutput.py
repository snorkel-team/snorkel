import argparse
import codecs 
from bs4 import BeautifulSoup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
		    description="Script to extract words coordinates in an xml file")
    parser.add_argument("input_html", help="html file to parse")
    parser.add_argument("ouput_ids", help="text file containing unique ids and corresponding word")
    parser.add_argument("ouput_coord", help="text file containing unique ids and coordinates (x,y,height,width)")
    parser.add_argument("pdf_page_number")
    parser.add_argument("--delimiter", help="delimiter in output files", default="\t") 
    args = parser.parse_args()
    html_in = codecs.open(args.input_html, encoding='utf-8').read()
    ids_words = codecs.open(args.ouput_ids, 'w', encoding='utf-8')
    ids_coordinates = codecs.open(args.ouput_coord, 'w', encoding='utf-8')
    page_nb = args.pdf_page_number
    delimiter = args.delimiter
    soup = BeautifulSoup(html_in, "html.parser")
    words = soup.find_all("word")
    i = 0
    for word in words:
        xmin = word.get('xmin')
	xmax = word.get('xmax')
	ymin = word.get('ymin')
	ymax = word.get('ymax')
	content = word.getText()
	ids_words.write((page_nb + str(i) + delimiter + content + '\n'))
	ids_coordinates.write(delimiter.join([page_nb + str(i), page_nb, ymin, xmin, ymax, xmax ]) + '\n')
		i +=1
    ids_words.close()
    ids_coordinates.close()

