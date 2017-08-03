import codecs
import glob
import os
import re
import lxml.etree as et

from bs4 import BeautifulSoup

from ..models import Document


class DocPreprocessor(object):
    """
    Processes a file or directory of files into a set of Document objects.

    :param encoding: file encoding to use, default='utf-8'
    :param path: filesystem path to file or directory to parse
    :param max_docs: the maximum number of Documents to produce,
        default=float('inf')

    """

    def __init__(self, path, encoding="utf-8", max_docs=float('inf')):
        self.path = path
        self.encoding = encoding
        self.max_docs = max_docs

    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.

        """
        doc_count = 0
        for fp in self._get_files(self.path):
            file_name = os.path.basename(fp)
            if self._can_read(file_name):
                for doc, text in self.parse_file(fp, file_name):
                    yield doc, text
                    doc_count += 1
                    if doc_count >= self.max_docs:
                        return

    def __iter__(self):
        return self.generate()

    def get_stable_id(self, doc_id):
        return "%s::document:0:0" % doc_id

    def parse_file(self, fp, file_name):
        raise NotImplementedError()

    def _can_read(self, fpath):
        return True

    def _get_files(self, path):
        if os.path.isfile(path):
            fpaths = [path]
        elif os.path.isdir(path):
            fpaths = [os.path.join(path, f) for f in os.listdir(path)]
        else:
            fpaths = glob.glob(path)
        if len(fpaths) > 0:
            return fpaths
        else:
            raise IOError("File or directory not found: %s" % (path,))


class TSVDocPreprocessor(DocPreprocessor):
    """Simple parsing of TSV file with one (doc_name <tab> doc_text) per line"""

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as tsv:
            for line in tsv:
                (doc_name, doc_text) = line.split('\t')
                stable_id = self.get_stable_id(doc_name)
                doc = Document(
                    name=doc_name, stable_id=stable_id,
                    meta={'file_name': file_name}
                )
                yield doc, doc_text


class TextDocPreprocessor(DocPreprocessor):
    """Simple parsing of raw text files, assuming one document per file"""

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            name = os.path.basename(fp).rsplit('.', 1)[0]
            stable_id = self.get_stable_id(name)
            doc = Document(
                name=name, stable_id=stable_id, meta={'file_name': file_name}
            )
            yield doc, f.read()


class CSVPathsPreprocessor(DocPreprocessor):
    """This `DocumentPreprocessor` treats inputs file as index of paths to
     actual documents; each line in the input file contains a path to a document.

     **Defaults and Customization:**

     * The input file is treated as a simple text file having one path per file.
       However, if the input is a CSV file, a pair of ``column`` and ``delim``
       parameters may be used to retrieve the desired value as reference path.

     * The referenced documents are treated as text document and hence parsed
       using ``TextDocPreprocessor``. However, if the referenced files are
       complex, an advanced parser may be used by specifying ``parser_factory``
       parameter to constructor.
     """

    def __init__(self, path, parser_factory=TextDocPreprocessor, column=None,
                 delim=',', *args, **kwargs):
        """
        :param path: input file having paths
        :param parser_factory: The parser class to be used to parse the
            referenced files. default = TextDocPreprocessor
        :param column: index of the column which references path.
                 default=None, which implies that each line has only one column
        :param delim: delimiter to be used to separate columns when file has
                      more than one column. It is active only when
                      ``column is not None``. default=','
        """
        super(CSVPathsPreprocessor, self).__init__(path, *args, **kwargs)
        self.column = column
        self.delim = delim
        self.parser = parser_factory(path)

    def _get_files(self, path):
        with codecs.open(path, encoding=self.encoding) as lines:
            for doc_path in lines:
                if self.column is not None:
                    # if column is set, retrieve specific column from CSV record
                    doc_path = doc_path.split(self.delim)[self.column]
                yield doc_path.strip()

    def parse_file(self, fp, file_name):
        return self.parser.parse_file(fp, file_name)


class TikaPreprocessor(DocPreprocessor):
    """
    This preprocessor use `Apache Tika <http://tika.apache.org>`_ parser to 
    retrieve text content from complex file types such as DOCX, HTML and PDFs.

    Documentation for customizing Tika is 
    `here <https://github.com/chrismattmann/tika-python>`_

    Example::

        !find pdf_dir -name *.pdf > input.csv # list of files
        from snorkel.parser import (
            TikaPreprocessor, CSVPathsPreprocessor, CorpusParser
        )
        CorpusParser().apply(
            CSVPathsPreprocessor('input.csv', parser_factory=TikaPreprocessor)
        )
    """
    # Tika is conditionally imported here
    import tika
    # automatically downloads tika jar and starts a JVM processif no REST API
    # is configured in ENV
    tika.initVM()  
    from tika import parser as tk_parser
    parser = tk_parser

    def parse_file(self, fp, file_name):
        parsed = type(self).parser.from_file(fp)
        txt = parsed['content']
        name = os.path.basename(fp).rsplit('.', 1)[0]
        stable_id = self.get_stable_id(name)
        doc = Document(
            name=name, stable_id=stable_id, meta={'file_name': file_name}
        )
        yield doc, txt


class HTMLDocPreprocessor(DocPreprocessor):
    """Simple parsing of raw HTML files, assuming one document per file"""

    def parse_file(self, fp, file_name):
        with open(fp, 'rb') as f:
            html = BeautifulSoup(f, 'lxml')
            txt = filter(self._cleaner, html.findAll(text=True))
            txt = ' '.join(self._strip_special(s) for s in txt if s != '\n')
            name = os.path.basename(fp).rsplit('.', 1)[0]
            stable_id = self.get_stable_id(name)
            doc = Document(
                name=name, stable_id=stable_id, meta={'file_name': file_name}
            )
            yield doc, txt

    def _can_read(self, fpath):
        return fpath.endswith('.html')

    def _cleaner(self, s):
        if s.parent.name in ['style', 'script', '[document]', 'head', 'title']:
            return False
        elif re.match('<!--.*-->', unicode(s)):
            return False
        return True

    def _strip_special(self, s):
        return (''.join(c for c in s if ord(c) < 128)).encode('ascii', 'ignore')


class XMLMultiDocPreprocessor(DocPreprocessor):
    """
    Parse an XML file _which contains multiple documents_ into a set of Document
    objects.

    Use XPath queries to specify a _document_ object, and then for each
    document, a set of _text_ sections and an _id_.

    **Note: Include the full document XML etree in the attribs dict with
    keep_xml_tree=True**
    """

    def __init__(self, path, doc='.//document', text='./text/text()',
        id='./id/text()', keep_xml_tree=False, *args, **kwargs):
        super(XMLMultiDocPreprocessor, self).__init__(path, *args, **kwargs)
        self.doc = doc
        self.text = text
        self.id = id
        self.keep_xml_tree = keep_xml_tree

    def parse_file(self, f, file_name):
        for i, doc in enumerate(et.parse(f).xpath(self.doc)):
            doc_id = str(doc.xpath(self.id)[0])
            text = '\n'.join(
                filter(lambda t: t is not None, doc.xpath(self.text))
            )
            meta = {'file_name': str(file_name)}
            if self.keep_xml_tree:
                meta['root'] = et.tostring(doc)
            stable_id = self.get_stable_id(doc_id)
            yield Document(name=doc_id, stable_id=stable_id, meta=meta), text

    def _can_read(self, fpath):
        return fpath.endswith('.xml')
