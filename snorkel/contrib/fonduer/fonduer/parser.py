# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import codecs
from collections import defaultdict
import itertools
import os
import re
import warnings

from lxml import etree
from lxml.html import fromstring
import numpy as np

from ....models import Candidate, Context, Document, construct_stable_id, split_stable_id
from .models import Table, Cell, Phrase, Figure, Para, Section, Header, FigureCaption, TableCaption, RefList

from ....udf import UDF, UDFRunner
from .visual import VisualLinker

from ....parser import DocPreprocessor, StanfordCoreNLPServer


class HTMLPreprocessor(DocPreprocessor):
    """Simple parsing of files into html documents"""
    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            soup = BeautifulSoup(f, 'lxml')
            for text in soup.find_all('html'):
                name = os.path.basename(fp)[:os.path.basename(fp).rfind('.')]
                stable_id = self.get_stable_id(name)
                import sys
                reload(sys)
                sys.setdefaultencoding('utf8')
                yield Document(name=name, stable_id=stable_id, text=unicode(text),
                               meta={'file_name' : file_name}), unicode(text)

    def _can_read(self, fpath):
        return fpath.endswith('html')  # includes both .html and .xhtml


class SimpleTokenizer(object):
    """
    A trivial alternative to CoreNLP which parses (tokenizes) text on
    whitespace only using the split() command.
    """
    def __init__(self, delim):
        self.delim = delim

    def parse(self, document, contents):
        i = 0
        for text in contents.split(self.delim):
            if not len(text.strip()):
                continue
            words = text.split()
            char_offsets = [0] + list(np.cumsum(map(lambda x: len(x) + 1, words)))[:-1]
            text = ' '.join(words)
            stable_id = construct_stable_id(document, 'phrase', i, i)
            yield {'text': text,
                   'words': words,
                   'char_offsets': char_offsets,
                   'stable_id': stable_id}
            i += 1


class OmniParser(UDFRunner):
    def __init__(self,
                 structural=True,                    # structural
                 blacklist=["style"],
                 flatten=['span', 'br'],
                 flatten_delim='',
                 lingual=True,                       # lingual
                 strip=True,
                 replacements=[],#[(u'[\u2010\u2011\u2012\u2013\u2014\u2212\uf02d]', '-')],
                 tabular=True,                       # tabular
                 visual=False,
                 pdf_path=None):

        self.delim = "<NB>"  # NB = New Block

        self.lingual_parser = StanfordCoreNLPServer(annotator_opts={"tokenize":{"normalizeSpace": False,"normalizeFractions":False,"normalizeParentheses":False,"normalizeOtherBrackets":False,"normalizeCurrency":False,"asciiQuotes": False,"latexQuotes": False, "unicodeQuotes": False, "ptb3Ellipsis": False, "unicodeEllipsis": False, "ptb3Dashes": False,"escapeForwardSlashAsterisk": False,"strictTreebank3": True}}, delimiter=self.delim[1:-1])

        super(OmniParser, self).__init__(OmniParserUDF,
                                         structural=structural,
                                         blacklist=blacklist,
                                         flatten=flatten,
                                         flatten_delim=flatten_delim,
                                         lingual=lingual, strip=strip,
                                         replacements=replacements,
                                         tabular=tabular,
                                         visual=visual,
                                         pdf_path=pdf_path,
                                         lingual_parser=self.lingual_parser)


    def clear(self, session, **kwargs):
        session.query(Context).delete()

        # We cannot cascade up from child contexts to parent Candidates, so we delete all Candidates too
        session.query(Candidate).delete()


class OmniParserUDF(UDF):
    def __init__(self,
                 structural,              # structural
                 blacklist,
                 flatten,
                 flatten_delim,
                 lingual,                 # lingual
                 strip,
                 replacements,
                 tabular,                 # tabular
                 visual,                  # visual
                 pdf_path,
                 lingual_parser,
                 **kwargs):
        """
        :param visual: boolean, if True visual features are used in the model
        :param pdf_path: directory where pdf are saved, if a pdf file is not found,
        it will be created from the html document and saved in that directory
        :param replacements: a list of (_pattern_, _replace_) tuples where _pattern_ isinstance
        a regex and _replace_ is a character string. All occurents of _pattern_ in the
        text will be replaced by _replace_.
        """
        super(OmniParserUDF, self).__init__(**kwargs)

        self.delim = "<NB>"  # NB = New Block

        # structural (html) setup
        self.structural = structural
        self.blacklist = blacklist if isinstance(blacklist, list) else [blacklist]
        self.flatten = flatten if isinstance(flatten, list) else [flatten]
        self.flatten_delim = flatten_delim

        # lingual setup
        self.lingual = lingual
        self.strip = strip
        self.replacements = []
        for (pattern, replace) in replacements:
            self.replacements.append((re.compile(pattern, flags=re.UNICODE), replace))
        if self.lingual:
            self.batch_size = 7000  # TODO(bhancock8): what if this is smaller than a block?
            self.lingual_parser = lingual_parser
            self.req_handler = lingual_parser.connect()
            self.lingual_parse = self.req_handler.parse

        else:
            self.batch_size = int(1e6)
            self.lingual_parse = SimpleTokenizer(delim=self.delim).parse

        # tabular setup
        self.tabular = tabular

        # visual setup
        self.visual = visual
        if self.visual:
            self.pdf_path = pdf_path
            self.vizlink = VisualLinker()

    def apply(self, x, **kwargs):
        document, text = x
        if self.visual:
            if not self.pdf_path:
                warnings.warn("Visual parsing failed: pdf_path is required", RuntimeWarning)
            for _ in self.parse_structure(document, text):
                pass
            # Add visual attributes
            filename = self.pdf_path + document.name
            create_pdf = not os.path.isfile(filename + '.pdf') and not os.path.isfile(filename + '.PDF') and not os.path.isfile(filename)
            if create_pdf:  # PDF file does not exist
                self.vizlink.create_pdf(document.name, text)
            for phrase in self.vizlink.parse_visual(document.name, document.phrases, self.pdf_path):
                yield phrase
        else:
            for phrase in self.parse_structure(document, text):
                yield phrase

    def parse_structure(self, document, text):
        self.contents = ""
        block_lengths = []
        self.parent = document

        figure_info = FigureInfo(document, parent=document)
        self.figure_idx = -1

        para_info = ParaInfo(document, parent=document)
        self.para_idx = -1
        parents_para = []

        section_info = SectionInfo(document, parent=document)
        self.section_idx = -1
        parents_section = []

        header_info = HeaderInfo(document, parent=document)
        self.header_idx = -1
        parents_header = []

        figCaption_info = FigureCaptionInfo(document, parent=document)
        self.figCaption_idx = -1
        parents_figCaption = []

        tabCaption_info = TableCaptionInfo(document, parent=document)
        self.tabCaption_idx = -1
        parents_tabCaption = []

        refList_info = RefListInfo(document, parent=document)
        self.refList_idx = -1
        parents_refList = []

        self.coordinates = {}
        self.char_idx = {}
        
        if self.structural:
            xpaths = []
            html_attrs = []
            html_tags = []

        if self.tabular:
            table_info = TableInfo(document, parent=document)
            self.table_idx = -1
            parents = []
        else:
            table_info = None

        def flatten(node):
            # if a child of this node is in self.flatten, construct a string
            # containing all text/tail results of the tree based on that child
            # and append that to the tail of the previous child or head of node
            num_children = len(node)
            for i, child in enumerate(node[::-1]):
                if child.tag in self.flatten:
                    j = num_children - 1 - i  # child index walking backwards
                    contents = ['']
                    for descendant in child.getiterator():
                        if descendant.text and descendant.text.strip():
                            contents.append(descendant.text)
                        if descendant.tail and descendant.tail.strip():
                            contents.append(descendant.tail)
                    if j == 0:
                        if node.text is None:
                            node.text = ''
                        node.text += self.flatten_delim.join(contents)
                    else:
                        if node[j - 1].tail is None:
                            node[j - 1].tail = ''
                        node[j - 1].tail += self.flatten_delim.join(contents)
                    node.remove(child)

        def parse_node(node, table_info=None, figure_info=None, para_info=None, section_info=None, header_info=None, figCaption_info=None, tabCaption_info=None, refList_info=None):
            if node.tag is etree.Comment:
                return
            if self.blacklist and node.tag in self.blacklist:
                return

            self.para_idx, coordinates = para_info.enter_para(node, self.para_idx, {})
            if len(coordinates)>0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0
            self.figure_idx = figure_info.enter_figure(node, self.figure_idx)
            self.section_idx, self.para_idx, coordinates = section_info.enter_section(node, self.section_idx, self.para_idx, {})
            if len(coordinates)>0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0
            self.header_idx, self.para_idx, coordinates = header_info.enter_header(node, self.header_idx, self.para_idx, {})
            if len(coordinates)>0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0
            self.figCaption_idx, self.para_idx, coordinates = figCaption_info.enter_figCaption(node, self.figCaption_idx, self.para_idx, {})
            if len(coordinates)>0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0
            self.tabCaption_idx, self.para_idx, coordinates = tabCaption_info.enter_tabCaption(node, self.tabCaption_idx, self.para_idx, {})
            if len(coordinates)>0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0
            self.refList_idx, self.para_idx, coordinates = refList_info.enter_refList(node, self.refList_idx, self.para_idx, {})
            if len(coordinates)>0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0

            if self.tabular:
                self.table_idx, self.para_idx, coordinates = table_info.enter_tabular(node, self.table_idx, self.para_idx, {})
                if len(coordinates)>0:
                    self.coordinates[self.para_idx] = coordinates
                    self.char_idx[self.para_idx] = 0

            if self.flatten:
                flatten(node)  # flattens children of node that are in the 'flatten' list

            for field in ['text', 'tail']:
                text = getattr(node, field)
                if text is not None:
                    if self.strip:
                        text = text.strip()
                    if len(text):
                        for (rgx, replace) in self.replacements:
                            text = rgx.sub(replace, text)
                        self.contents += text
                        self.contents += self.delim
                        block_lengths.append(len(text) + len(self.delim))

                        parents_para.append(para_info.parent)
                        parents_section.append(section_info.parent)
                        parents_header.append(header_info.parent)
                        parents_figCaption.append(figCaption_info.parent)
                        parents_tabCaption.append(tabCaption_info.parent)
                        parents_refList.append(refList_info.parent)

                        if self.tabular:
                            parents.append(table_info.parent)

                        if self.structural:
                            context_node = node.getparent() if field == 'tail' else node
                            xpaths.append(tree.getpath(context_node))
                            html_tags.append(context_node.tag)
                            html_attrs.append(map(lambda x: '='.join(x), context_node.attrib.items()))

            for child in node:
                if child.tag == 'table':
                    parse_node(child, TableInfo(document=table_info.document), figure_info, para_info, section_info, header_info, figCaption_info, tabCaption_info, refList_info)
                elif child.tag == 'figure':
                    parse_node(child, table_info, FigureInfo(document=figure_info.document), para_info, section_info, header_info, figCaption_info, tabCaption_info, refList_info)
                elif child.tag == 'paragraph':
                    parse_node(child, table_info, figure_info, ParaInfo(document=para_info.document), section_info, header_info, figCaption_info, tabCaption_info, refList_info)
                elif child.tag == 'section_header':
                    parse_node(child, table_info, figure_info, para_info, SectionInfo(document=section_info.document), header_info, figCaption_info, tabCaption_info, refList_info)
                elif child.tag == 'header':
                    parse_node(child, table_info, figure_info, para_info, section_info, HeaderInfo(document=header_info.document), figCaption_info, tabCaption_info, refList_info)
                elif child.tag == 'figure_caption':
                    parse_node(child, table_info, figure_info, para_info, section_info, header_info, FigureCaptionInfo(document=figCaption_info.document), tabCaption_info, refList_info)
                elif child.tag == 'table_caption':
                    parse_node(child, table_info, figure_info, para_info, section_info, header_info, figCaption_info, TableCaptionInfo(document=tabCaption_info.document), refList_info)
                elif child.tag == 'list':
                    parse_node(child, table_info, figure_info, para_info, section_info, header_info, figCaption_info, tabCaption_info, RefListInfo(document=refList_info.document))
                else:
                    parse_node(child, table_info, figure_info, para_info, section_info, header_info, figCaption_info, tabCaption_info, refList_info)

            if self.tabular:
                table_info.exit_tabular(node)

            refList_info.exit_refList(node)
            tabCaption_info.exit_tabCaption(node)
            figCaption_info.exit_figCaption(node)
            header_info.exit_header(node)
            section_info.exit_section(node)
            para_info.exit_para(node)
            figure_info.exit_figure(node)

            
        # Parse document and store text in self.contents, padded with self.delim
        # import sys
        # reload(sys)
        # sys.setdefaultencoding('utf8')
        root = fromstring(text)#.decode('utf-8'))  # lxml.html.fromstring()
        tree = etree.ElementTree(root)
        document.text = text#.decode('utf-8')
        parse_node(root, table_info, figure_info, para_info, section_info, header_info, figCaption_info, tabCaption_info, refList_info)
        block_char_end = np.cumsum(block_lengths)
        content_length = len(self.contents)
        parsed = 0
        parent_idx = 0
        position = 0
        phrase_num = 0
        abs_phrase_offset = 0
        while parsed < content_length:
            batch_end = parsed + \
                        self.contents[parsed:parsed + self.batch_size].rfind(self.delim) + \
                        len(self.delim)
            for parts in self.lingual_parse(document,self.contents[parsed:batch_end]):
                # print (self.contents[parsed:batch_end])
                (_, _, _, char_end) = split_stable_id(parts['stable_id'])
                try:
                    while parsed + char_end > block_char_end[parent_idx]:
                        parent_idx += 1
                        position = 0
                    parts['document'] = document
                    parts['phrase_num'] = phrase_num
                    abs_phrase_offset_end = abs_phrase_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
                    parts['stable_id'] = construct_stable_id(document, 'phrase', abs_phrase_offset, abs_phrase_offset_end)
                    abs_phrase_offset = abs_phrase_offset_end
                    if self.structural:
                        parts['xpath'] = xpaths[parent_idx]
                        parts['html_tag'] = html_tags[parent_idx]
                        parts['html_attrs'] = html_attrs[parent_idx]
                    parent = parents_para[parent_idx]
                    parts, self.char_idx = para_info.apply_para(parts, parent, position, self.coordinates, self.char_idx)
                    # print "here 2"
                    parent = parents_section[parent_idx]
                    parts, self.char_idx = section_info.apply_section(parts, parent, position, self.coordinates, self.char_idx)

                    parent = parents_header[parent_idx]
                    parts, self.char_idx = header_info.apply_header(parts, parent, position, self.coordinates, self.char_idx)

                    parent = parents_figCaption[parent_idx]
                    parts, self.char_idx = figCaption_info.apply_figCaption(parts, parent, position, self.coordinates, self.char_idx)

                    parent = parents_tabCaption[parent_idx]
                    parts, self.char_idx = tabCaption_info.apply_tabCaption(parts, parent, position, self.coordinates, self.char_idx)
                    

                    parent = parents_refList[parent_idx]
                    parts, self.char_idx = refList_info.apply_refList(parts, parent, position, self.coordinates, self.char_idx)

                    if self.tabular:
                        parent = parents[parent_idx]
                        parts = table_info.apply_tabular(parts, parent, position, self.coordinates)
                    yield Phrase(**parts)
                    position += 1
                    phrase_num += 1
                except Exception as e:
                    print("[ERROR]" + str(e))
                    import pdb
                    pdb.set_trace()
            parsed = batch_end

class TableInfo(object):
    def __init__(self, document,
                 table=None, table_grid=defaultdict(int),
                 cell=None, cell_idx=0,
                 row_idx=0, col_idx=0,
                 parent=None):
        self.document = document
        self.table = table
        self.table_grid = table_grid
        self.cell = cell
        self.cell_idx = cell_idx
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.parent = parent

    def enter_tabular(self, node, table_idx, para_idx, coordinates):
        if node.tag == "table":
            table_idx += 1
            self.table_grid.clear()
            self.row_idx = 0
            self.cell_position = 0
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "table", table_idx, table_idx)
            self.table = Table(document=self.document, stable_id=stable_id,
                                position=table_idx)
            self.parent = self.table
        elif node.tag == "tr":
            self.col_idx = 0
        elif node.tag in ["td", "th"]:
            # calculate row_start/col_start
            while self.table_grid[(self.row_idx, self.col_idx)]:
                self.col_idx += 1
            col_start = self.col_idx
            row_start = self.row_idx

            # calculate row_end/col_end
            row_end = row_start
            if "rowspan" in node.attrib:
                row_end += int(node.get("rowspan")) - 1
            col_end = col_start
            if "colspan" in node.attrib:
                col_end += int(node.get("colspan")) - 1

            # update table_grid with occupied cells
            for r, c in itertools.product(range(row_start, row_end + 1),
                                            range(col_start, col_end + 1)):
                self.table_grid[r, c] = 1

            # construct para
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_cell", para_idx, para_idx)
            self.para = Para(document=self.document, stable_id=stable_id,
                                position=para_idx)

            # construct cell
            parts = defaultdict(list)
            parts["document"] = self.document
            parts["table"] = self.table
            parts["para"] = self.para
            # parts["para_id"] = self.para.id
            parts["row_start"] = row_start
            parts["row_end"] = row_end
            parts["col_start"] = col_start
            parts["col_end"] = col_end
            parts["position"] = self.cell_position
            parts["stable_id"] = "%s::%s:%s:%s:%s" % \
                                    (self.document.name, "cell",
                                     self.table.position, row_start, col_start)
            self.cell = Cell(**parts)
            self.parent = self.cell
            #coordinates
            coordinates = {}
            coordinates["word"] = node.get('word')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return table_idx, para_idx, coordinates

    def exit_tabular(self, node):
        if node.tag == "table":
            self.table = None
            self.parent = self.document
        elif node.tag == "tr":
            self.row_idx += 1
        elif node.tag in ["td", "th"]:
            self.cell = None
            self.col_idx += 1
            self.cell_idx += 1
            self.cell_position += 1
            self.parent = self.table

    def apply_tabular(self, parts, parent, position, coordinates):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Table):
            parts['table'] = parent
        elif isinstance(parent, Cell):
            parts['table'] = parent.table
            parts['cell'] = parent
            parts['para'] = parent.para
            parts['row_start'] = parent.row_start
            parts['row_end'] = parent.row_end
            parts['col_start'] = parent.col_start
            parts['col_end'] = parent.col_end
            parts = update_coordinates_table(parts, coordinates[parent.para.position])
        else:
            raise NotImplementedError("Phrase parent must be Document, Table, or Cell")
        return parts

class FigureInfo(object):
    def __init__(self, document,
                 figure=None, parent=None):
        self.document = document
        self.figure = figure
        self.parent = parent

    def enter_figure(self, node, figure_idx):
        if node.tag == "figure":
            figure_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "figure", figure_idx, figure_idx)
            self.figure = Figure(document=self.document, stable_id=stable_id,
                                    position=figure_idx)
            self.parent = self.figure
        return figure_idx

    def exit_figure(self, node):
        if node.tag == "figure":
            self.figure = None
            self.parent = self.document

class ParaInfo(object):
    def __init__(self, document,
                 para=None, parent=None):
        self.document = document
        self.para = para
        self.parent = parent

    def enter_para(self, node, para_idx, coordinates):
        if node.tag == "paragraph":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para", para_idx, para_idx)
            self.para = Para(document=self.document, stable_id=stable_id,
                                position=para_idx)
            self.parent = self.para
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return para_idx, coordinates

    def exit_para(self, node):
        if node.tag == "paragraph":
            self.para = None
            self.parent = self.document
        
    def apply_para(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Para):
            parts['para'] = parent
            parts, char_idx[parent.position] = update_coordinates(parts, coordinates[parent.position], char_idx[parent.position])
        else:
            raise NotImplementedError("Phrase parent must be Document or Para")
        return parts, char_idx

def update_coordinates_table(parts, coordinates):
    sep = " "
    words = coordinates["word"][:-1].split(sep)
    top = [float(_) for _ in coordinates["top"][:-1].split(sep)]
    left = [float(_) for _ in coordinates["left"][:-1].split(sep)]
    bottom = [float(_) for _ in coordinates["bottom"][:-1].split(sep)]
    right = [float(_) for _ in coordinates["right"][:-1].split(sep)]
    max_len = len(words)
    i=0
    for word in parts["words"]:
        parts['top'].append(top)
        parts['left'].append(left)
        parts['bottom'].append(bottom)
        parts['right'].append(right)
        i += 1
        if i == max_len:
            break	
    return parts

def lcs(X , Y):
    m = len(X)
    n = len(Y)
 
    L = [[None]*(n+1) for i in xrange(m+1)]
    d = [[None]*(n+1) for i in xrange(m+1)]

    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    matches = []
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1] and (L[i-1][j-1]+1)>max(L[i-1][j], L[i][j-1]):
                L[i][j] = L[i-1][j-1]+1
                d[i][j] = 'd'
            else:
                if L[i][j-1] > L[i-1][j]:
                    d[i][j] = 'u'
                    L[i][j] = L[i][j-1]
                else:
                    d[i][j] = 'l'
                    L[i][j] = L[i-1][j]
    i = m
    j = n
    while i>=0 and j>=0:
        if d[i][j] == 'u':
            j -= 1
        elif d[i][j] == 'l':
            i -= 1
        else:
            matches.append((i,j))
            i -= 1
            j -= 1
    return matches

def update_coordinates(parts, coordinates, char_idx):
    sep = " "
    chars = coordinates["char"][:-1].split(sep)
    top = [float(_) for _ in coordinates["top"][:-1].split(sep)]
    left = [float(_) for _ in coordinates["left"][:-1].split(sep)]
    bottom = [float(_) for _ in coordinates["bottom"][:-1].split(sep)]
    right = [float(_) for _ in coordinates["right"][:-1].split(sep)]
    words = []
    new_chars = []
    new_top = []
    new_left = []
    new_bottom = []
    new_right = []
    for i, char in enumerate(chars):
        if len(char) > 0:
            new_chars.append(char)
            new_top.append(top[i])
            new_left.append(left[i])
            new_bottom.append(bottom[i])
            new_right.append(right[i])
    chars = new_chars
    top = new_top
    left = new_left
    right = new_right
    bottom = new_bottom
    words = []
    #print "".join(chars)
    matches = lcs("".join(chars[char_idx:]), "".join(parts["words"]))
    word_lens = [len(words) for words in parts["words"]]
    for i, word in enumerate(parts["words"]):
        curr_word = [word, float("Inf"), float("Inf"), float("-Inf"), float("-Inf")]
        word_len = 0
        word_len += sum(word_lens[:i])
        word_begin = -1
        word_end = -1
        for match in matches:
            if match[1] == word_len:
                word_begin = match[0]
            if match[1] == word_len + word_lens[i]:
                word_end = match[0]
        if word_begin == -1 or word_end == -1:
            print "no match found"
        else:
            for char_iter in range(word_begin, word_end):
                curr_word[1] = int(min(curr_word[1], top[char_idx+char_iter]))
                curr_word[2] = int(min(curr_word[2], left[char_idx+char_iter]))
                curr_word[3] = int(max(curr_word[3], bottom[char_idx+char_iter]))
                curr_word[4] = int(max(curr_word[4], right[char_idx+char_iter]))        
        parts['top'].append(curr_word[1])
        parts['left'].append(curr_word[2])
        parts['bottom'].append(curr_word[3])
        parts['right'].append(curr_word[4])
    #print char_idx, max([x[0] for x in matches])
    char_idx += max([x[0] for x in matches])
    
    '''
    for word in parts["words"]:
        curr_word = [word, float("Inf"), float("Inf"), float("-Inf"), float("-Inf")]
        len_idx = 0
        while len_idx<len(word):
            while word[len_idx] == " ":
                len_idx += 1
            if chars[char_idx].decode("utf-8") == u'\u204e':
                char_idx += 1
            if word[len_idx]!=chars[char_idx].replace('"',"'") and word[len_idx]!=chars[char_idx]:
                print "Out of order", word, word[len_idx], chars[char_idx]
                len_idx += 1
            else:
                curr_word[1] = min(curr_word[1], top[char_idx])
                curr_word[2] = min(curr_word[2], left[char_idx])
                curr_word[3] = max(curr_word[3], bottom[char_idx])
                curr_word[4] = max(curr_word[4], right[char_idx])
                len_idx += len(chars[char_idx])
                char_idx += 1
        words.append(curr_word)
        parts['top'].append(curr_word[1])
        parts['left'].append(curr_word[2])
        parts['bottom'].append(curr_word[3])
        parts['right'].append(curr_word[4])
    '''
    return parts, char_idx

class SectionInfo(object):
    def __init__(self, document,
                 section=None, parent=None):
        self.document = document
        self.section = section
        self.parent = parent

    def enter_section(self, node, section_idx, para_idx, coordinates):
        if node.tag == "section_header":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_section", para_idx, para_idx)
            para = Para(document=self.document, stable_id=stable_id,
                                position=para_idx)

            section_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "section", section_idx, section_idx)
            self.section = Section(document=self.document, stable_id=stable_id,
                                position=section_idx, para=para, para_id=para_idx)
            self.parent = self.section
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return section_idx, para_idx, coordinates

    def exit_section(self, node):
        if node.tag == "section_header":
            self.section = None
            self.parent = self.document

    def apply_section(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Section):
            parts['para'] = parent.para
            parts, char_idx[parent.para.position] = update_coordinates(parts, coordinates[parent.para.position], char_idx[parent.para.position])
        else:
            raise NotImplementedError("Phrase parent must be Document or Section")
        return parts, char_idx

class HeaderInfo(object):
    def __init__(self, document,
                 header=None, parent=None):
        self.document = document
        self.header = header
        self.parent = parent

    def enter_header(self, node, header_idx, para_idx, coordinates):
        if node.tag == "header":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_header", para_idx, para_idx)
            para = Para(document=self.document, stable_id=stable_id,
                                position=para_idx)

            header_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "header", header_idx, header_idx)
            self.header = Header(document=self.document, stable_id=stable_id,
                                position=header_idx, para=para)
            self.parent = self.header
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return header_idx, para_idx, coordinates

    def exit_header(self, node):
        if node.tag == "header":
            self.header = None
            self.parent = self.document

    def apply_header(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Header):
            parts['para'] = parent.para
            parts, char_idx[parent.para.position] = update_coordinates(parts, coordinates[parent.para.position], char_idx[parent.para.position])
        else:
            raise NotImplementedError("Phrase parent must be Document or Header")
        return parts, char_idx

class FigureCaptionInfo(object):
    def __init__(self, document,
                 figCaption=None, parent=None):
        self.document = document
        self.figCaption = figCaption
        self.parent = parent

    def enter_figCaption(self, node, figCaption_idx, para_idx, coordinates):
        if node.tag == "figure_caption":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_figCaption", para_idx, para_idx)
            para = Para(document=self.document, stable_id=stable_id,
                                position=para_idx)

            figCaption_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "figCaption", figCaption_idx, figCaption_idx)
            self.figCaption = FigureCaption(document=self.document, stable_id=stable_id,
                                position=figCaption_idx, para=para)
            self.parent = self.figCaption
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return figCaption_idx, para_idx, coordinates

    def exit_figCaption(self, node):
        if node.tag == "figure_caption":
            self.figCaption = None
            self.parent = self.document

    def apply_figCaption(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, FigureCaption):
            parts['para'] = parent.para
            parts, char_idx[parent.para.position] = update_coordinates(parts, coordinates[parent.para.position], char_idx[parent.para.position])
        else:
            raise NotImplementedError("Phrase parent must be Document or FigureCaption")
        return parts, char_idx

class TableCaptionInfo(object):
    def __init__(self, document,
                 tabCaption=None, parent=None):
        self.document = document
        self.tabCaption = tabCaption
        self.parent = parent

    def enter_tabCaption(self, node, tabCaption_idx, para_idx, coordinates):
        if node.tag == "table_caption":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_tabCaption", para_idx, para_idx)
            para = Para(document=self.document, stable_id=stable_id,
                                position=para_idx)

            tabCaption_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "tabCaption", tabCaption_idx, tabCaption_idx)
            self.tabCaption = TableCaption(document=self.document, stable_id=stable_id,
                                position=tabCaption_idx, para=para)
            self.parent = self.tabCaption
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return tabCaption_idx, para_idx, coordinates

    def exit_tabCaption(self, node):
        if node.tag == "table_caption":
            self.tabCaption = None
            self.parent = self.document

    def apply_tabCaption(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, TableCaption):
            parts['para'] = parent.para
            parts, char_idx[parent.para.position] = update_coordinates(parts, coordinates[parent.para.position], char_idx[parent.para.position])
        else:
            raise NotImplementedError("Phrase parent must be Document or TableCaption")
        return parts, char_idx

class RefListInfo(object):
    def __init__(self, document,
                 refList=None, parent=None):
        self.document = document
        self.refList = refList
        self.parent = parent

    def enter_refList(self, node, refList_idx, para_idx, coordinates):
        if node.tag == "list":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_refList", para_idx, para_idx)
            para = Para(document=self.document, stable_id=stable_id,
                                position=para_idx)

            refList_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "refList", refList_idx, refList_idx)
            self.refList = RefList(document=self.document, stable_id=stable_id,
                                position=refList_idx, para=para)
            self.parent = self.refList
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return refList_idx, para_idx, coordinates

    def exit_refList(self, node):
        if node.tag == "list":
            self.refList = None
            self.parent = self.document

    def apply_refList(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, RefList):
            parts['para'] = parent.para
            parts, char_idx[parent.para.position] = update_coordinates(parts, coordinates[parent.para.position], char_idx[parent.para.position])
        else:
            raise NotImplementedError("Phrase parent must be Document or RefList")
        return parts, char_idx
