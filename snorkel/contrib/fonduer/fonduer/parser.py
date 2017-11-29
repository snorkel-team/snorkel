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
from .models import Table, Cell, Figure, Phrase

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
                 structural=True,                    # structural information
                 blacklist=["style"],                # ignore tag types, default: style
                 flatten=['span', 'br'],             # flatten tag types, default: span, br
                 flatten_delim='',
                 lingual=True,                       # lingual information
                 strip=True,
                 replacements=[(u'[\u2010\u2011\u2012\u2013\u2014\u2212\uf02d]', '-')],
                 tabular=True,                       # tabular information
                 visual=False,                       # visual information
                 pdf_path=None):

        self.delim = "<NB>"  # NB = New Block

        self.lingual_parser = StanfordCoreNLPServer(delimiter=self.delim[1:-1])

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
            self.batch_size = 7000
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
            create_pdf = not os.path.isfile(filename + '.pdf') and not os.path.isfile(filename + '.PDF')
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

        def parse_node(node, table_info=None, figure_info=None):
            if node.tag is etree.Comment:
                return
            if self.blacklist and node.tag in self.blacklist:
                return

            self.figure_idx = figure_info.enter_figure(node, self.figure_idx)

            if self.tabular:
                self.table_idx = table_info.enter_tabular(node, self.table_idx)

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

                        if self.tabular:
                            parents.append(table_info.parent)

                        if self.structural:
                            context_node = node.getparent() if field == 'tail' else node
                            xpaths.append(tree.getpath(context_node))
                            html_tags.append(context_node.tag)
                            html_attrs.append(map(lambda x: '='.join(x), context_node.attrib.items()))

            for child in node:
                if child.tag == 'table':
                    parse_node(child, TableInfo(document=table_info.document), figure_info)
                elif child.tag == 'img':
                    parse_node(child, table_info, FigureInfo(document=figure_info.document))
                else:
                    parse_node(child, table_info, figure_info)

            if self.tabular:
                table_info.exit_tabular(node)

            figure_info.exit_figure(node)

        # Parse document and store text in self.contents, padded with self.delim
        root = fromstring(text)  # lxml.html.fromstring()
        tree = etree.ElementTree(root)
        document.text = text
        parse_node(root, table_info, figure_info)
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
            for parts in self.lingual_parse(document,
                                            self.contents[parsed:batch_end]):
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
                    if self.tabular:
                        parent = parents[parent_idx]
                        parts = table_info.apply_tabular(parts, parent, position)
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

    def enter_tabular(self, node, table_idx):
        if node.tag == "table":
            table_idx += 1
            self.table_grid.clear()
            self.row_idx = 0
            self.cell_position = 0
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "table", table_idx, table_idx)
            self.table = Table(document=self.document, stable_id=stable_id, position=table_idx)
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

            # construct cell
            parts = defaultdict(list)
            parts["document"] = self.document
            parts["table"] = self.table
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
        return table_idx

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

    def apply_tabular(self, parts, parent, position):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Table):
            parts['table'] = parent
        elif isinstance(parent, Cell):
            parts['table'] = parent.table
            parts['cell'] = parent
            parts['row_start'] = parent.row_start
            parts['row_end'] = parent.row_end
            parts['col_start'] = parent.col_start
            parts['col_end'] = parent.col_end
        else:
            raise NotImplementedError("Phrase parent must be Document, Table, or Cell")
        return parts


class FigureInfo(object):
    def __init__(self, document, figure=None, parent=None):
        self.document = document
        self.figure = figure
        self.parent = parent

    def enter_figure(self, node, figure_idx):
        if node.tag == "img":
            figure_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "figure", figure_idx, figure_idx)
            self.figure = Figure(document=self.document, stable_id=stable_id, position=figure_idx, url=node.get('src'))
            self.parent = self.figure
        return figure_idx

    def exit_figure(self, node):
        if node.tag == "img":
            self.figure = None
            self.parent = self.document
