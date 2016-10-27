import numpy as np
from collections import OrderedDict
from editdistance import eval as editdist
import subprocess
import os
import cv2

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


def display_box(pdf_path, boxes, page_width=612, page_height=792):
    """
    Displays each of the bounding boxes passed in 'boxes' on the pdf pointed 
    to by pdf_path
    """
    # boxes is a list of 5-tuples (page, top, left, bottom, right)
    basename = subprocess.check_output('basename {} .pdf'.format(pdf_path), shell=True)
    dirname = subprocess.check_output('dirname {}'.format(pdf_path), shell=True)
    im_path = dirname.rstrip() + '/' + basename.rstrip()
    page_nb = boxes[0][0] # take the page number of the first box in the list
    os.system('pdftoppm -f {} -l {} -jpeg {} {}'.format(page_nb, page_nb, pdf_path, im_path))
    im_path += '-{}.jpg'.format(page_nb)
    img = cv2.resize(cv2.imread(im_path),(page_width,page_height))
    # Plot bounding boxes
    for p, top, left, bottom, right in boxes:
	    if not (p == page_nb):
		    raise Exception("Can not display bounding boxes that are not in the same page.")
	    cv2.rectangle(img,(int(float(left)),int(float(top))),(int(float(right)),int(float(bottom))),(255,0,0),1)
    cv2.imshow('Bounding boxes',img)
    cv2.waitKey() # press any key to exit the opencv output 
    cv2.destroyAllWindows() 
    # delete image
    os.system('rm {}'.format(im_path))


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
