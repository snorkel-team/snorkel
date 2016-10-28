import subprocess
import os
import cv2
import numpy as np
from bs4 import BeautifulSoup
from pandas import DataFrame, Series
from collections import OrderedDict, defaultdict
from editdistance import eval as editdist
from snorkel.models import Phrase

class VisualLinker():
    def __init__(self, session):
        self.session = session

    def extract_pdf_words(self, pdf_file):
        num_pages = subprocess.check_output(
            "pdfinfo {} | grep Pages  | sed 's/[^0-9]*//'".format(pdf_file), shell=True)
        pdf_word_list = []
        coordinate_map= {}
        for i in range(1, int(num_pages)+1):
            html_content = subprocess.check_output('pdftotext -f {} -l {} -bbox-layout {} -'.format(str(i), str(i), pdf_file), shell=True)
            pdf_word_list_i, coordinate_map_i = self._coordinates_from_HTML(html_content, i)
            # TODO: this is a hack for testing; use a more permanent solution for tokenizing
            pdf_word_list_additions = []
            for j, (word_id, word) in enumerate(pdf_word_list_i):
                if not word[-1].isalnum():
                    pdf_word_list_i[j] = (word_id, word[:-1])
                    page, idx = word_id
                    new_word_id = (page, idx + 0.5)
                    pdf_word_list_additions.append((new_word_id, word[-1]))
                    coordinate_map_i[new_word_id] = coordinate_map_i[word_id]
            pdf_word_list_i.extend(pdf_word_list_additions)
            # sort pdf_word_list by page, then top, then left
            pdf_word_list += sorted(pdf_word_list_i, key=lambda (word_id,_): coordinate_map_i[word_id][0:3])
            coordinate_map.update(coordinate_map_i)
        self.pdf_file = pdf_file
        self.pdf_word_list = pdf_word_list
        self.coordinate_map = coordinate_map
        print "Extracted %d pdf words" % len(self.pdf_word_list)

    def _coordinates_from_HTML(self, html_content, page_num):
        pdf_word_list = []
        coordinate_map = {}
        soup = BeautifulSoup(html_content, "html.parser")
        lines = soup.find_all('line')
        i = 0  # counter for word_id in page_num
        for line in lines:
            j = 0  # counter for words within a line (words that are not whitespaces)
            words = line.find_all("word")
            y_min_line = int(float(words[0].get('ymin')))
            y_max_line = 0
            for word in words:
                xmin = int(float(word.get('xmin')))
                xmax = int(float(word.get('xmax')))
                ymin = int(float(word.get('ymin')))
                ymax = int(float(word.get('ymax')))
                content = word.getText()
                if len(content) > 0:  # Ignore white spaces
                    if ymin < y_min_line:
                        y_min_line = ymin
                    if ymax > y_max_line:
                        y_max_line = ymax
                    word_id = (page_num, i)
                    pdf_word_list.append((word_id, content))
                    coordinate_map[word_id] = (page_num, xmin, xmax)  # TODO: check this order
                    i += 1
                    j +=1
            for word_id, _ in pdf_word_list[-j:]:
                page_num, xmin, xmax = coordinate_map[word_id]
                coordinate_map[word_id] = (page_num, y_min_line, xmin, y_max_line, xmax)
        return pdf_word_list, coordinate_map

    def extract_html_words(self, document):
        self.document = document
        html_word_list = []
        for phrase in document.phrases:
            for i, word in enumerate(phrase.words):
                html_word_list.append(((phrase.id, i), word))
        self.html_word_list = html_word_list
        print "Extracted %d html words" % len(self.html_word_list)

    def link_better(self):
        N = len(self.html_word_list)
        links = [None] * N
        # make dicts of word -> id
        html_dict = defaultdict(list)
        pdf_dict = defaultdict(list)
        for i, (uid, word) in enumerate(self.html_word_list):
            html_dict[word].append((uid, i))
        for j, (uid, word) in enumerate(self.pdf_word_list):
            pdf_dict[word].append((uid, j))
        for word, html_list in html_dict.items():
            if len(html_list) == len(pdf_dict[word]):
                pdf_list = pdf_dict[word]
                for j, (uid, i) in enumerate(id_list):
                    links[i] = pdf_list[j][0]
        # what percent of links are None?

        # convert list to dict
        self.links = OrderedDict((self.html_word_list[i][0], links[i]) for i in range(N))


    def link_lists(self, searchMax=200, editCost=20, offsetCost=1, offsetInertia=5):
        DEBUG = True
        if DEBUG:
            offsetHist = []
            jHist = []
            editDistHist = 0
        offset = self._calculate_offset(self.html_word_list, self.pdf_word_list, max(searchMax/10,5), searchMax)
        offsets = [offset] * offsetInertia
        searchOrder = np.array([(-1)**(i%2) * (i/2) for i in range(1, searchMax+1)])
        links = OrderedDict()
        for i, a in enumerate(self.html_word_list):
            j = 0
            searchIndices = np.clip(offset + searchOrder, 0, len(self.pdf_word_list)-1)
            jMax = len(searchIndices)
            matched = False
            # Search first for exact matches
            while not matched and j < jMax:
                b = self.pdf_word_list[searchIndices[j]]
                if a[1] == b[1]:
                    links[a[0]] = b[0]
                    matched = True
                    offsets[i % offsetInertia] = searchIndices[j]  + 1
                    offset = int(np.median(offsets))
                    if DEBUG:
                        jHist.append(j)
                        offsetHist.append(offset)
                j += 1
            # If necessary, search for min edit distance
            if not matched:
                cost = [0] * searchMax
                for k, m in enumerate(searchIndices):
                    cost[k] = (editdist(a[1],self.pdf_word_list[m][1]) * editCost +
                            k * offsetCost)
                nearest = np.argmin(cost)            
                links[a[0]] = self.pdf_word_list[searchIndices[nearest]][0]
                if DEBUG:
                    jHist.append(nearest)
                    offsetHist.append(searchIndices[nearest])
                    editDistHist += 1
        if DEBUG:
            # print offsetHist
            # print jHist
            # print editDistHist
            self.offsetHist = offsetHist
        self.links = links
        print "Linked %d words to %d bounding boxes" % (len(self.html_word_list), len(self.pdf_word_list))
        self.update_coordinates()
        print "Updated coordinates in snorkel.db"

    def _calculate_offset(self, listA, listB, seedSize, maxOffset):
        wordsA = zip(*listA[:seedSize])[1]
        wordsB = zip(*listB[:maxOffset])[1]
        offsets = []
        for i in range(seedSize):
            try:
                offsets.append(wordsB.index(wordsA[i]) - i)
            except:
                pass
        return int(np.median(offsets))

    def display_links(self):
        html = []
        pdf = []
        j = []
        for i, l in enumerate(self.links):
            html.append(self.html_word_list[i][1])
            for k, b in enumerate(self.pdf_word_list):
                if b[0] == self.links[self.html_word_list[i][0]]:
                    pdf.append(b[1])
                    j.append(k)
                    break
        assert(len(pdf) == len(html))

        data = {
            'i': range(len(self.links)),
            'html': html,
            'pdf': pdf,
            'j': j,
            'offset': self.offsetHist
        }
        return DataFrame(data, columns=['i','html','pdf','j','offset'])

    def update_coordinates(self):
        for phrase in self.document.phrases:
            (page, top, left, bottom, right) = zip(
                *[self.coordinate_map[self.links[((phrase.id), i)]] for i in range(len(phrase.words))])
            page = page[0]
            self.session.query(Phrase).filter(Phrase.id == phrase.id).update(
                {"page": page, 
                "top":  top, 
                "left": left, 
                "bottom": bottom, 
                "right": right})
        self.session.commit()

    def display_candidates(self, candidates, page_num=1, display=True, page_width=612, page_height=792):
        """
        Displays the bounding boxes corresponding to candidates on an image of the pdf
        # boxes is a list of 5-tuples (page, top, left, bottom, right)
        """
        if display:
            (img, img_path) = pdf_to_img(self.pdf_file, page_num, page_width, page_height)
            colors = [(0,0,255), (255,0,0)]
        boxes_by_page = defaultdict(list)
        drawn = 0
        not_drawn = 0
        total = 0
        for c in candidates:
            for i, span in enumerate(c.get_arguments()):
                page, top, left, bottom, right = get_box(span)
                if page == page_num and (top, left, bottom, right) not in boxes_by_page[page]:
                    drawn += 1
                    if display:
                        cv2.rectangle(img, (left, top), (right, bottom), colors[i], 1)
                else:
                    not_drawn += 1
                boxes_by_page[page].append((top, left, bottom, right))
                total += 1
        print "Drawn: %d" % drawn
        print "Not drawn: %d" % not_drawn
        print "Total: %d" % total
        print "Boxes per page:"
        for (page, boxes) in sorted(boxes_by_page.items()):
            print "Page %d: %d (%d)" % (page, len(boxes), len(set(boxes)))
        if display:
            cv2.imshow('Bounding boxes', img)
            cv2.waitKey() # press any key to exit the opencv output 
            cv2.destroyAllWindows() 
            os.system('rm {}'.format(img_path)) # delete image


def pdf_to_img(pdf_file, page_num, page_width, page_height):
    basename = subprocess.check_output('basename {} .pdf'.format(pdf_file), shell=True)
    dirname = subprocess.check_output('dirname {}'.format(pdf_file), shell=True)
    img_path = dirname.rstrip() + '/' + basename.rstrip()
    os.system('pdftoppm -f {} -l {} -jpeg {} {}'.format(page_num, page_num, pdf_file, img_path))
    img_path += '-{}.jpg'.format(page_num)
    img = cv2.resize(cv2.imread(img_path), (page_width, page_height))
    return (img, img_path)

def get_box(span):
    return (span.parent.page, 
            min(span.get_attrib_tokens('top')),
            max(span.get_attrib_tokens('left')),
            min(span.get_attrib_tokens('bottom')),
            max(span.get_attrib_tokens('right')))

def display_boxes(pdf_file, boxes, page_num=1, page_width=612, page_height=792):
    """
    Displays each of the bounding boxes passed in 'boxes' on an image of the pdf
    pointed to by pdf_file
    # boxes is a list of 5-tuples (page, top, left, bottom, right)
    """
    (img, img_path) = pdf_to_img(pdf_file, page_num, page_width, page_height)
    boxes_per_page = defaultdict(int)
    for page, top, left, bottom, right in boxes:
        if page == page_num:
            cv2.rectangle(img, (left, top), (right, bottom), (255,0,0), 1)
        boxes_per_page[page] += 1
    print "Boxes per page:"
    for (page, count) in sorted(boxes_per_page.items()):
        print "Page %d: %d" % (page, count)
    cv2.imshow('Bounding boxes', img)
    cv2.waitKey() # press any key to exit the opencv output 
    cv2.destroyAllWindows() 
    os.system('rm {}'.format(img_path)) # delete image

