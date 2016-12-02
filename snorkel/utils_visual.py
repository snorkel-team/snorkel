import subprocess
from bs4 import BeautifulSoup
from collections import defaultdict
from wand.image import Image
from wand.drawing import Drawing
from wand.display import display
from wand.color import Color


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


def display_candidates(candidates, page_num=1, display=True):
    """
    Displays the bounding boxes corresponding to candidates on an image of the pdf
    boxes is a list of 5-tuples (page, top, left, bottom, right)
    """
    pass


def display_words(target=None, page_num=1, display=True):
    pass


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
