from .models import Phrase, TemporarySpan
import subprocess
from collections import namedtuple
from bs4 import BeautifulSoup
from collections import defaultdict
from wand.image import Image
from wand.drawing import Drawing
from wand.display import display
from wand.color import Color


Bbox = namedtuple('bbox', ['page', 'top', 'bottom', 'left', 'right'], verbose=False)


def bbox_from_span(span):
    if isinstance(span, TemporarySpan) and span.is_visual():
        return Bbox(
                span.get_attrib_tokens('page')[0],
                min(span.get_attrib_tokens('top')),
                max(span.get_attrib_tokens('bottom')),
                min(span.get_attrib_tokens('left')),
                max(span.get_attrib_tokens('right')))
    else:
        return None


def bbox_from_phrase(phrase):
    # TODO: this may have issues where a phrase is linked to words on different pages
    if isinstance(phrase, Phrase) and phrase.is_visual():
        return Bbox(
                phrase.page[0],
                min(phrase.top),
                max(phrase.bottom),
                min(phrase.left),
                max(phrase.right))
    else:
        return None


def bbox_horz_aligned(box1, box2):
    """
    Returns true if the vertical center point of either span is within the 
    vertical range of the other
    """
    if not (box1 and box2): return False
    center1 = (box1.bottom + box1.top) / 2.0
    center2 = (box2.bottom + box2.top) / 2.0
    return ((center1 >= box2.top and center1 <= box2.bottom) or
            (center2 >= box1.top and center2 <= box1.bottom))


def bbox_vert_aligned(box1, box2):
    """
    Returns true if the horizontal center point of either span is within the 
    horizontal range of the other
    """
    if not (box1 and box2): return False
    center1 = (box1.right + box1.left) / 2.0
    center2 = (box2.right + box2.left) / 2.0
    return ((center1 >= box2.left and center1 <= box2.right) or
            (center2 >= box1.left and center2 <= box1.right))


def bbox_vert_aligned_left(box1, box2):
    """
    Returns true if the left boundary of both boxes is within 2 pts
    """
    if not (box1 and box2): return False
    return abs(box1.left - box2.left) <= 2


def bbox_vert_aligned_right(box1, box2):
    """
    Returns true if the right boundary of both boxes is within 2 pts
    """
    if not (box1 and box2): return False
    return abs(box1.right - box2.right) <= 2


def bbox_vert_aligned_center(box1, box2):
    """
    Returns true if the right boundary of both boxes is within 5 pts
    """
    if not (box1 and box2): return False
    return abs((box1.right + box1.left) / 2.0 - (box2.right + box2.left) / 2.0) <= 5


### DISPLAY TOOLS


def display_boxes(pdf_file, boxes, page_num=1, display_img=True, alternate_colors=False):
    """
    Displays each of the bounding boxes passed in 'boxes' on an image of the pdf
    pointed to by pdf_file
    boxes is a list of 5-tuples (page, top, left, bottom, right)
    """
    if display_img:
        img = pdf_to_img(pdf_file, page_num)
        colors = [Color('blue'), Color('red')]
    boxes_per_page = defaultdict(int)
    boxes_by_page = defaultdict(list)
    for i, (page, top, left, bottom, right) in enumerate(boxes):
        boxes_per_page[page] += 1
        boxes_by_page[page].append((top, left, bottom, right))
    if display_img:
        draw = Drawing()
        draw.fill_color = Color('rgba(0, 0, 0, 0.0)')
        for j, (top, left, bottom, right) in enumerate(boxes_by_page[page_num]):
            draw.stroke_color = colors[j % 2] if alternate_colors else colors[0]
            draw.rectangle(left=left, top=top, right=right, bottom=bottom)
        draw(img)
    print "Boxes per page: total (unique)"
    for (page, count) in sorted(boxes_per_page.items()):
        print "Page %d: %d (%d)" % (page, count, len(set(boxes_by_page[page])))
    if display_img:
        display(img)


def get_pdf_dim(pdf_file):
    html_content = subprocess.check_output(
            "pdftotext -f {} -l {} -bbox '{}' -".format('1', '1', pdf_file), shell=True)
    soup = BeautifulSoup(html_content, "html.parser")
    pages = soup.find_all('page')
    page_width, page_height = int(float(pages[0].get('width'))), int(float(pages[0].get('height')))
    return page_width, page_height


def pdf_to_img(pdf_file, page_num, pdf_dim=None):
    """
    Converts pdf file into image
    :param pdf_file: path to the pdf file
    :param page_num: page number to convert (index starting at 1)
    :return: wand image object
    """
    if not pdf_dim:
        pdf_dim = get_pdf_dim(pdf_file)
    page_width, page_height = pdf_dim
    img = Image(filename='{}[{}]'.format(pdf_file, page_num - 1))
    img.resize(page_width, page_height)
    return img


def get_box(span):
    box = (min(span.get_attrib_tokens('page')),
           min(span.get_attrib_tokens('top')),
           max(span.get_attrib_tokens('left')),
           min(span.get_attrib_tokens('bottom')),
           max(span.get_attrib_tokens('right')))
    return box


def display_candidates(pdf_file, candidates, page_num=1, display=True):
    """
    Displays the bounding boxes corresponding to candidates on an image of the pdf
    boxes is a list of 5-tuples (page, top, left, bottom, right)
    """
    boxes = [get_box(span) for c in candidates for span in c.get_arguments()]
    display_boxes(pdf_file, boxes, page_num=page_num, display_img=display, alternate_colors=True)


def display_words(pdf_file, phrases, target=None, page_num=1, display=True):
    boxes = []
    for phrase in phrases:
        for i, word in enumerate(phrase.words):
            if target is None or word == target:
                boxes.append((
                    phrase.page[i],
                    phrase.top[i],
                    phrase.left[i],
                    phrase.bottom[i],
                    phrase.right[i]))
    display_boxes(pdf_file, boxes, page_num=page_num, display_img=display)
