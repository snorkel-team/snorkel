import subprocess
import os
import cv2
import numpy as np
from bs4 import BeautifulSoup
from collections import OrderedDict, defaultdict
from editdistance import eval as editdist
from snorkel.models import Phrase

def extract_coordinates(pdf_file):
    num_pages = subprocess.check_output(
        "pdfinfo {} | grep Pages  | sed 's/[^0-9]*//'".format(pdf_file), shell=True)
    pdf_word_list = []
    coordinate_map= {}
    for i in range(1, int(num_pages)+1):
        html_content = subprocess.check_output('pdftotext -f {} -l {} -bbox-layout {} -'.format(str(i), str(i), pdf_file), shell=True)
        pdf_word_list_i, coordinate_map_i = _coordinates_from_HTML(html_content, i)
        # sort pdf_word_list by page, then top, then left
        pdf_word_list += sorted(pdf_word_list_i, key=lambda (word_id,_): coordinate_map_i[word_id][0:3])
        coordinate_map.update(coordinate_map_i)
    return pdf_word_list, coordinate_map

def _coordinates_from_HTML(html_content, page_num):
    pdf_word_list = []
    coordinate_map= {}
    soup = BeautifulSoup(html_content, "html.parser")
    words = soup.find_all("word")
    i = 0
    for word in words:
        xmin = int(float(word.get('xmin')))
        xmax = int(float(word.get('xmax')))
        ymin = int(float(word.get('ymin')))
        ymax = int(float(word.get('ymax')))
        content = word.getText()
        if len(content) > 0: # Ignore white spaces
            word_id = (page_num, i)
            pdf_word_list.append((word_id, content))
            coordinate_map[word_id] = (page_num, ymin, xmin, ymax, xmax) #TODO: check this order
            i += 1
    return pdf_word_list, coordinate_map


def extract_words(corpus):
    html_word_list = []
    for phrase in corpus.documents[0].phrases:
        for i, word in enumerate(phrase.words):
            html_word_list.append(((phrase.id, i), word))
    return html_word_list


def load_coordinates(corpus, links, coordinate_map, session):
    for phrase in corpus.documents[0].phrases:
        (page, top, left, bottom, right) = zip(
            *[coordinate_map[links[((phrase.id), i)]] for i in range(len(phrase.words))])
        page = page[0]
        session.query(Phrase).filter(Phrase.id == phrase.id).update(
            {"page": page, 
             "top":  top, 
             "left": left, 
             "bottom": bottom, 
             "right": right})
    session.commit()

def calculate_offset(listA, listB, seedSize, maxOffset):
    wordsA = zip(*listA[:seedSize])[1]
    wordsB = zip(*listB[:maxOffset])[1]
    offsets = []
    for i in range(seedSize):
        try:
            offsets.append(wordsB.index(wordsA[i]) - i)
        except:
            pass
    return int(np.median(offsets))

def get_box(span):
    return (span.parent.page, 
            span.parent.top[0], 
            span.parent.left[0], 
            span.parent.bottom[0],
            span.parent.right[0])

def pdf_to_img(pdf_file, page_num, page_width, page_height):
    basename = subprocess.check_output('basename {} .pdf'.format(pdf_file), shell=True)
    dirname = subprocess.check_output('dirname {}'.format(pdf_file), shell=True)
    img_path = dirname.rstrip() + '/' + basename.rstrip()
    os.system('pdftoppm -f {} -l {} -jpeg {} {}'.format(page_num, page_num, pdf_file, img_path))
    img_path += '-{}.jpg'.format(page_num)
    img = cv2.resize(cv2.imread(img_path), (page_width, page_height))
    return (img, img_path)


def display_candidates(pdf_file, candidates, page_num=1, page_width=612, page_height=792):
    """
    Displays the bounding boxes corresponding to candidates on an image of the pdf
    pointed to by pdf_file
    # boxes is a list of 5-tuples (page, top, left, bottom, right)
    """
    (img, img_path) = pdf_to_img(pdf_file, page_num, page_width, page_height)
    colors = [(0,0,255), (255,0,0)]
    boxes_by_page = defaultdict(list)
    for c in candidates:
        for i, span in enumerate(c.get_arguments()):
            page, top, left, bottom, right = get_box(span)
            if page == page_num:
                cv2.rectangle(img, (left, top), (right, bottom), colors[i], 1)
            boxes_by_page[page] += (top, left, bottom, right)
    print "Boxes per page:"
    for (page, boxes) in sorted(boxes_by_page.items()):
        print "Page %d: %d (%d)" % (page, len(boxes), len(set(boxes)))
    cv2.imshow('Bounding boxes', img)
    cv2.waitKey() # press any key to exit the opencv output 
    cv2.destroyAllWindows() 
    os.system('rm {}'.format(img_path)) # delete image



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



def link_lists(listA, listB, searchMax=100, editCost=20, offsetCost=1, offsetInertia=5):
    DEBUG = False
    if DEBUG:
        offsetHist = []
        jHist = []
        editDistHist = 0
    offset = calculate_offset(listA, listB, max(searchMax/10,5), searchMax)
    offsets = [offset] * offsetInertia
    searchOrder = np.array([(-1)**(i%2) * (i/2) for i in range(1, searchMax+1)])
    links = OrderedDict()
    for i, a in enumerate(listA):
        j = 0
        searchIndices = np.clip(offset + searchOrder, 0, len(listB)-1)
        jMax = len(searchIndices)
        matched = False
        # Search first for exact matches
        while not matched and j < jMax:
            b = listB[searchIndices[j]]
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
                cost[k] = (editdist(a[1],listB[m][1]) * editCost +
                           k * offsetCost)
            links[a[0]] = listB[searchIndices[np.argmin(cost)]][0]
            if DEBUG:
                editDistHist += 1
    if DEBUG:
        print offsetHist
        print jHist
        print editDistHist
    return links
