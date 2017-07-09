import os
import re
import sys
import glob
import codecs
import shutil
import signal
import zipfile
import tarfile
import itertools
import subprocess

from sqlalchemy import and_
from .utils import download
from collections import defaultdict
from IPython.display import IFrame, display, HTML
from ...models import Span, Candidate, Document, Sentence, TemporarySpan, GoldLabel, GoldLabelKey
from ...learning.utils import print_scores

class BratAnnotator(object):
    """
    Snorkel Interface fo
    Brat Rapid Annotation Tool
    http://brat.nlplab.org/

    This implements a minimal interface for annotating simple relation pairs and their entities.

    """
    def __init__(self, session, candidate_class, encoding="utf-8",
                 annotator_name='brat', address='localhost', port=8001):
        """
        Begin BRAT session by:
        - checking that all app files are downloaded
        - creating/validate a local file system mirror of documents
        - launch local server

        :param session:
        :param candidate_class:
        :param address:
        :param port:
        """
        self.session = session
        self.candidate_class = candidate_class
        self.address = address
        self.port = port
        self.encoding = encoding

        self.path = os.path.dirname(os.path.realpath(__file__))
        self.brat_root = 'brat-v1.3_Crunchy_Frog'
        self.data_root = "{}/{}/data".format(self.path, self.brat_root)

        self.standoff_parser = StandoffAnnotations(encoding=self.encoding)

        # setup snorkel annotator object
        self.annotator = self.session.query(GoldLabelKey).filter(GoldLabelKey.name == annotator_name).first()
        if self.annotator is None:
            self.annotator = GoldLabelKey(name=annotator_name)
            self.session.add(self.annotator)
            self.session.commit()

        self._download()
        self.process_group = None
        self._start_server()

    def init_collection(self, annotation_dir, split=None, cid_query=None,
                        overwrite=False, errors='replace'):
        """
        Initialize document collection on disk

        :param doc_root:
        :param split:
        :param cid_query:
        :param overwrite:
        :return:
        """
        assert split != None or cid_query != None

        collection_path = "{}/{}".format(self.data_root, annotation_dir)
        if os.path.exists(collection_path) and not overwrite:
            msg = "Error! Collection at '{}' already exists. ".format(annotation_dir)
            msg += "Please set overwrite=True to erase all existing annotations.\n"
            sys.stderr.write(msg)
            return

        # remove existing annotations
        if os.path.exists(collection_path):
            shutil.rmtree(collection_path, ignore_errors=True)
            print("Removed existing collection at '{}'".format(annotation_dir))

        # create subquery based on candidate split
        if split != None:
            cid_query = self.session.query(Candidate.id).filter(Candidate.split == split).subquery()

        # generate all documents for this candidate set
        doc_ids = get_doc_ids_by_query(self.session, self.candidate_class, cid_query)
        documents = self.session.query(Document).filter(Document.id.in_(doc_ids)).all()

        # create collection on disk
        os.makedirs(collection_path)

        for doc in documents:
            text = doc_to_text(doc)
            outfpath = "{}/{}".format(collection_path, doc.name)
            with codecs.open(outfpath + ".txt","w", self.encoding, errors=errors) as fp:
                fp.write(text)
            with codecs.open(outfpath + ".ann","w", self.encoding, errors=errors) as fp:
                fp.write("")

        # add minimal annotation.config based on candidate_subclass info
        self._init_annotation_config(self.candidate_class, annotation_dir)

    def import_collection(self, zip_archive, overwrite=False):
        """
        Import zipped archive of BRAT documents and annotations.
        NOTE zip file must preserve full directory structure.

        :param archive:
        :param overwrite:
        :return:
        """
        out_dir = "{}/".format(self.data_root)
        zip_ref = zipfile.ZipFile(zip_archive, 'r')
        manifest = zip_ref.namelist()
        if not manifest:
            msg = "ERROR: Zipfile is empty. Nothing to import"
            sys.stderr.write(msg)
            return

        if os.path.exists(out_dir + manifest[0]) and not overwrite:
            fpath = out_dir + manifest[0]
            msg = "Error! Collection at '{}' already exists. ".format(fpath)
            msg += "Please set overwrite=True to erase all existing annotations.\n"
            sys.stderr.write(msg)
            return

        zip_ref.extractall(out_dir)
        zip_ref.close()
        print("Imported archive to {}".format(out_dir))

        # cleanup for files compressed on MacOS
        if os.path.exists(out_dir + "__MACOSX"):
            shutil.rmtree(out_dir + "__MACOSX")

    def view(self, annotation_dir, document=None, new_window=True):
        """
        Launch web interface for Snorkel. The default mode launches a new window.
        This is preferred as we have limited control of default widget sizes,
        which can cause display issues when rendering embedded in a Jupyter notebook cell.

        If no document is provided, we create a browser link to the file view mode of BRAT.
        Otherwise we create a link directly to the provided document

        :param document:
        :param new_window:
        :return:

        :param doc_root:
        :param document:
        :param new_window:
        :return:
        """
        # http://localhost:8001/index.xhtml#/pain/train/
        doc_name = document.name if document else ""
        url = "http://{}:{}/index.xhtml#/{}/{}".format(self.address, self.port, annotation_dir, doc_name)

        if new_window:
            # NOTE: if we use javascript, we need pop-ups enabled for a given browser
            #html = "<script>window.open('{}','_blank');</script>".format(url)
            html = "<a href='{}' target='_blank'>Launch BRAT</a>".format(url)
            display(HTML(html))

        else:
            self.display(url)

    def display(self, url, width='100%', height=700):
        """
        Create embedded iframe view of BRAT

        :param width:
        :param height:
        :return:
        """
        display(HTML("<style>.container { width:100% !important; }</style>"))
        display(IFrame(url, width=width, height=height))

    def map_annotations(self, session, annotation_dir, candidates, symmetric_relations=True):
        """
        Import a collection of BRAT annotations,  map it onto the provided set
        of candidates, and create gold labels. This method DOES NOT create new
        candidates, so some labels may not import if a corresponding candidate
        cannot be found.

        Enable show_errors to print out specific details on missing candidates.

        :param: session:
        :param doc_root:
        :param candidates:
        :param symmetric_relations: Boolean indicating whether to extract symmetric
                                    Candidates, i.e., rel(A,B) and rel(B,A), where
                                    A and B are Contexts. Only applies to binary
                                    relations. Default is True.
        :return:
        """
        # load BRAT annotations
        fpath = self.get_collection_path(annotation_dir)
        annotations = self.standoff_parser.load_annotations(fpath)

        # load Document objects from session
        doc_names = [doc_name for doc_name in annotations if annotations[doc_name]]
        documents = session.query(Document).filter(Document.name.in_(doc_names)).all()
        documents = {doc.name:doc for doc in documents}

        # TODO: make faster!!
        # create stable IDs for all candidates
        candidate_stable_ids = {}
        for c in candidates:
            candidate_stable_ids[(c[0].get_stable_id(), c[1].get_stable_id())] = c

        # build BRAT span/relation objects
        brat_stable_ids = []
        for doc_name in documents:
            spans, relations = self._create_relations(documents[doc_name], annotations[doc_name])
            for key in relations:
                brat_stable_ids.append(tuple([r.get_stable_id() for r in relations[key]]))

        mapped_cands, missed = [], []
        for relation in brat_stable_ids:
            # swap arguments if this is a symmetric relation
            if symmetric_relations and relation not in candidate_stable_ids:
                relation = (relation[1],relation[0])
            # otherwise just test if this relation is in our candidate set
            if relation in candidate_stable_ids:
                mapped_cands.append(candidate_stable_ids[relation])
            else:
                missed.append(relation)

        n, N = len(mapped_cands), len(missed) + len(mapped_cands)
        p = len(mapped_cands)/ float(N)
        print>>sys.stderr,"Mapped {}/{} ({:2.0f}%) of BRAT labels to candidates".format(n,N,p*100)
        return mapped_cands, len(missed)

    def error_analysis(self, session, candidates, marginals, annotation_dir, b=0.5):
        """

        :param session:
        :param candidates:
        :param marginals:
        :param annotation_dir:
        :param b:
        :param set_unlabeled_as_neg:
        :return:
        """
        mapped_cands, missed = self.map_annotations(session, annotation_dir, candidates)
        doc_ids = {c.get_parent().document.id for c in mapped_cands}

        subset_cands = [c for c in candidates if c.get_parent().document.id in doc_ids]
        marginals = {c.id: marginals[i] for i, c in enumerate(candidates)}

        tp = [c for c in mapped_cands if marginals[c.id] > b]
        fn = [c for c in mapped_cands if marginals[c.id] <= b]
        fp = [c for c in candidates if marginals[c.id] > b and c not in mapped_cands]
        tn = [c for c in candidates if marginals[c.id] <= b and c not in mapped_cands]

        return tp, fp, tn, fn

    def score(self, session, candidates, marginals, annotation_dir,
                       b=0.5, recall_correction=True, symmetric_relations=True):
        """

        :param session:
        :param candidates:
        :param marginals:
        :param annotation_dir:
        :param b:
        :param symmetric_relations:
        :return:
        """
        mapped_cands, missed = self.map_annotations(session, annotation_dir, candidates,
                                                    symmetric_relations=symmetric_relations)

        # determine the full set of document names over which we compute our metrics
        docs = glob.glob("{}/*.txt".format(self.get_collection_path(annotation_dir)))
        doc_names = set([os.path.basename(fp).split(".")[0] for fp in docs])

        subset_cands = [c for c in candidates if c.get_parent().document.name in doc_names]
        marginals = {c.id:marginals[i] for i,c in enumerate(candidates)}

        y_true = [1 if c in mapped_cands else 0 for c in subset_cands]
        y_pred = [1 if marginals[c.id] > b else 0 for c in subset_cands]

        print y_true
        print y_pred

        missed = 0 if not recall_correction else missed
        title = "{} BRAT Scores ({} Documents)".format("Unadjusted" if not recall_correction else "Adjusted",
                                                       len(doc_names))
        return self._score(y_true, y_pred, missed, title)

    def get_collection_path(self, annotation_dir):
        """
        Return directory path of provided annotation set
        :param annotation_dir:
        :return:
        """
        return "{}/{}".format(self.data_root, annotation_dir)

    def import_gold_labels(self, session, annotation_dir, candidates,
                           symmetric_relations=True,  annotator_name='brat'):
        """
        We assume all candidates provided to this function are true instances
        :param session:
        :param candidates:
        :param annotator_name:
        :return:
        """
        mapped_cands, _ = self.map_annotations(session, annotation_dir, candidates, symmetric_relations)

        for c in mapped_cands:
            if self.session.query(GoldLabel).filter(and_(GoldLabel.key_id == self.annotator.id,
                                                         GoldLabel.candidate_id == c.id,
                                                         GoldLabel.value == 1)).all():
                continue
            label = GoldLabel(key=self.annotator, candidate=c, value=1)
            session.add(label)
        session.commit()

    def _score(self, y_true, y_pred, recall_correction=0, title='BRAT Scores'):
        """

        :param y_pred:
        :param recall_correction:
        :return:
        """
        tp = [1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1]
        fp = [1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1]
        tn = [1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0]
        fn = [1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0]
        tp, fp, tn, fn = sum(tp), sum(fp), sum(tn), sum(fn)

        print_scores(tp, fp, tn, fn + recall_correction, title=title)

    def _close(self):
        '''
        Kill the process group linked with this server.
        :return:
        '''
        print("Killing BRAT server [{}]...".format(self.process_group.pid))
        if self.process_group is not None:
            try:
                os.kill(self.process_group.pid, signal.SIGTERM)
            except Exception as e:
                sys.stderr.write('Could not kill BRAT server [{}] {}\n'.format(self.process_group.pid, e))

    def _start_server(self):
        """
        Launch BRAT server

        :return:
        """
        cwd = os.getcwd()
        os.chdir("{}/{}/".format(self.path, self.brat_root))
        cmd = ["python", "standalone.py", "{}".format(self.port)]
        self.process_group = subprocess.Popen(cmd, cwd=os.getcwd(), env=os.environ, shell=False )
        os.chdir(cwd)
        url = "http://{}:{}".format(self.address, self.port)
        print("Launching BRAT server at {} [pid={}]...".format(url, self.process_group.pid))

    def _download(self):
        """
        Download and install latest version of BRAT
        :return:
        """
        fname = "{}/{}".format(self.path, 'brat-v1.3_Crunchy_Frog.tar.gz')
        if os.path.exists("{}/{}/".format(self.path,self.brat_root)):
            return

        url = "http://weaver.nlplab.org/~brat/releases/brat-v1.3_Crunchy_Frog.tar.gz"
        print("Downloading BRAT [{}]...".format(url))
        download(url, fname)

        # install brat
        cwd = os.getcwd()
        os.chdir(self.path)
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()

        print("Installing BRAT...")
        # setup default username and passwords
        shutil.copyfile("install.sh", "{}/install.sh".format(self.brat_root))
        os.chdir("{}/{}".format(self.path, self.brat_root))
        subprocess.call(["./install.sh"])

        # cleanup
        os.chdir(cwd)
        os.remove(fname)

    def _login(self):
        """
        BRAT requires a user login in order to edit annotations. We do this
        automatically behind the scenes.

        TODO: Not yet! User must login manually.
        :return:
        """
        # TODO -- this requires some jquery/cookie magic to automatically handle logins.
        pass

    def _init_annotation_config(self, candidate_class, doc_root):
        """

        :param candidate_class:
        :return:
        """
        collection_path = "{}/{}".format(self.data_root, doc_root)
        # create config file
        config = self.standoff_parser._create_config([candidate_class])
        config_path = "{}/annotation.conf".format(collection_path)
        with codecs.open(config_path, 'w', self.encoding) as fp:
            fp.write(config)

        # initalize tools config (this sets sentence tokenzation for visuzliation)
        shutil.copyfile("{}/templates/tools.conf".format(self.path),
                        "{}/tools.conf".format(collection_path))

    def _create_temp_span(self, document, abs_char_start, abs_char_end):
        """
        Given parsed snorkel document object and global, absolute char offsets,
        create a temporary span object.

        :param doc:
        :param abs_char_start:
        :param abs_char_end:
        :return:
        """
        sent_id = len(document.sentences) - 1
        sent_offsets = [sent.abs_char_offsets[0] for sent in document.sentences]
        for i in range(0, len(sent_offsets) - 1):
            if abs_char_start >= sent_offsets[i] and abs_char_end <= sent_offsets[i + 1]:
                sent_id = i
                break

        sent = document.sentences[sent_id]
        char_start = abs_char_start - sent.abs_char_offsets[0]
        char_end = abs_char_end - sent.abs_char_offsets[0]

        return TemporarySpan(sent, char_start, char_end - 1)

    def _create_relations(self, document, annotations):
        """
        Initalize temporary Span objects for all our named entity labels
        and then create lists of Span pairs for each relation label.

        :return:
        """
        # create span (entity) objects
        spans = {}
        for key in annotations:
            if key[0] != StandoffAnnotations.TEXT_BOUND_ID:
                continue
            i, j = annotations[key]["abs_char_start"], annotations[key]["abs_char_end"]
            mention = annotations[key]["mention"]
            spans[key] = self._create_temp_span(document, i, j)
            if spans[key].get_span() != mention:
                msg = "Warning: {} Span annotations do not match BRAT:[{}]!=SNORKEL:[{}] [{}:{}]".format(
                    document.name, mention, spans[key].get_span(), i, j
                )
                print >> sys.stderr, msg.format(key)

        # create relation pairs
        relations = {}
        for key in annotations:
            if key[0] != StandoffAnnotations.RELATION_ID:
                continue
            rtype, arg1, arg2 = annotations[key]
            # check that our span objects exist
            if arg1 not in spans or arg2 not in spans:
                msg = "Error: Relation {} missing Span object (check for Span parsing errors)"
                print >> sys.stderr, msg.format(key)
            relations[key] = [spans[arg1], spans[arg2]]

        return spans, relations

    def __del__(self):
        '''
        Clean-up this object by forcing the server process to shut-down
        :return:
        '''
        self._close()


class StandoffAnnotations(object):
    """
    Standoff Annotation Parser

    See:
        BioNLP Shared Task 2011     http://2011.bionlp-st.org/home/file-formats
        Brat Rapid Annotation Tool  http://brat.nlplab.org/standoff.html

    Annotation ID Types
    T: text-bound annotation
    R: relation
    E: event
    A: attribute
    M: modification (alias for attribute, for backward compatibility)
    N: normalization [new in v1.3]
    #: note

    Many of of the advanced schema abilities used by BRAT are not implemented, so
    mind the following caveats:

    (1) We do not currently support hierarchical entity definitions, e.g.,
            !Anatomical_entity
                !Anatomical_structure
                    Organism_subdivision
                    Anatomical_system
                    Organ
    (2) All relations must be binary with a single argument type
    (3) Attributes, normalization, and notes are added as candidate meta information

    """

    TEXT_BOUND_ID = 'T'
    RELATION_ID = 'R'
    EVENT_ID = 'E'
    ATTRIB_ID = 'A'
    MOD_ID = 'M'
    NORM_ID = 'N'
    NOTE_ID = '#'

    def __init__(self, tmpl_path='annotation.config.tmpl', encoding="utf-8"):
        """
        Initialize standoff annotation parser

        :param encoding:
        """
        self.encoding = encoding
        # load brat annotation config template
        mod_path = "{}/templates/{}".format(os.path.abspath(os.path.dirname(__file__)), tmpl_path)
        self.config_tmpl = "".join(open(mod_path, "rU").readlines())

    def load_annotations(self, input_dir):
        """
        Import BART project,
        :param input_dir:
        :param autoreload:
        :param num_threads:
        :param parser:
        :return:
        """
        config_path = "{}/{}".format(input_dir, "annotation.conf")
        if not os.path.exists(config_path):
            print>> sys.stderr, "Fatal error: missing 'annotation.conf' file"
            return

        # load brat config (this defines relation and argument types)
        config = self._parse_config(config_path)
        anno_filelist = set([os.path.basename(fn).strip(".ann") for fn in glob.glob(input_dir + "/*.ann")])

        # import standoff annotations for all documents
        annotations = {}
        for fn in anno_filelist:
            txt_fn = "{}/{}.txt".format(input_dir, fn)
            ann_fn = "{}/{}.ann".format(input_dir, fn)
            if os.path.exists(txt_fn) and os.path.exists(ann_fn):
                annotations[fn] = self._parse_annotations(txt_fn, ann_fn)

        return annotations

    def _parse_annotations(self, txt_filename, ann_filename):
        """
        Use parser to import BRAT standoff format

        :param txt_filename:
        :param ann_filename:
        :return:
        """
        annotations = {}

        # read document string
        with codecs.open(txt_filename, "rU", encoding=self.encoding) as fp:
            doc_str = fp.read()

        # load annotations
        with codecs.open(ann_filename, "rU", encoding=self.encoding) as fp:
            for line in fp:
                row = line.strip().split("\t")
                anno_id_prefix = row[0][0]

                # parse each entity/relation type
                if anno_id_prefix == StandoffAnnotations.TEXT_BOUND_ID:
                    anno_id, entity, text = row
                    entity_type = entity.split()[0]
                    spans = map(lambda x: map(int, x.split()), entity.lstrip(entity_type).split(";"))

                    # discontinuous mentions
                    if len(spans) != 1:
                        print>> sys.stderr, "NotImplementedError: Discontinuous spans"
                        continue

                    i,j = spans[0]
                    mention = doc_str[i:j]
                    # santity check to see if label span matches document span
                    if mention != text:
                        print>> sys.stderr, \
                            "Error: Annotation spans do not match {} != {}".format(mention, text)
                        continue

                    annotations[anno_id] = {"abs_char_start":i, "abs_char_end":j,
                                            "entity_type":entity_type, "mention":mention}

                elif anno_id_prefix in [StandoffAnnotations.RELATION_ID,'*']:
                    anno_id, rela = row
                    rela_type, arg1, arg2 = rela.split()
                    arg1 = arg1.split(":")[1] if ":" in arg1 else arg1
                    arg2 = arg2.split(":")[1] if ":" in arg2 else arg2
                    annotations[anno_id] = (rela_type, arg1, arg2)

                elif anno_id_prefix == StandoffAnnotations.EVENT_ID:
                    print>> sys.stderr, "NotImplementedError: Events"
                    raise NotImplementedError

                elif anno_id_prefix == StandoffAnnotations.ATTRIB_ID:
                    print>> sys.stderr, "NotImplementedError: Attributes"

        return annotations

    def _normalize_relation_name(self, name):
        """
        Normalize relation name

        :param name:
        :return:
        """
        name = re.split("[-_]", name)
        if len(name) == 1:
            return name[0]
        name = map(lambda x: x.lower(), name)
        return "".join(map(lambda x: x[0].upper() + x[1:], name))

    def _create_config(self, candidate_types):
        """
        Export a minimal BRAT configuration schema defining
        a binary relation and two argument types.

        :param candidate_type:
        :return:
        """
        entity_defs, rela_defs = [], []
        for stype in candidate_types:
            rel_type = str(stype.type).rstrip(".type")
            arg_types = [key.rstrip("_id") for key in stype.__dict__ if "_id" in key]
            arg_types = [name[0].upper()+name[1:] for name in arg_types]

            # HACK: Assume all args that differ by just a number are
            # of the same type, e.g., person1, person2
            arg_types = [re.sub("\d+$", "", name) for name in arg_types]

            entity_defs.extend(set(arg_types))
            if len(arg_types) > 1:
                rela_name = [str(stype.type).replace(".type","")] + arg_types
                rela_defs.append("{}\tArg1:{}, Arg2:{}".format(*rela_name))

        entity_defs = set(entity_defs)
        rela_defs = set(rela_defs)
        return self.config_tmpl.format("\n".join(entity_defs), "\n".join(rela_defs), "", "")

    def _parse_config(self, filename):
        """
        Parse BRAT annotation.config

        :param filename:
        :return:
        """
        config = defaultdict(list)
        with open(filename, "rU") as fp:
            curr = None
            for line in fp:
                # skip comments
                line = line.strip()
                if not line or line[0] == '#':
                    continue
                # brat definition?
                m = re.search("^\[(.+)\]$", line)
                if m:
                    curr = m.group(1)
                    continue
                config[curr].append(line)

        # type-specific parsing
        tmp = []
        for item in config['relations']:
            m = re.search("^(.+)\s+Arg1:(.+),\s*Arg2:(.+),*\s*(.+)*$", item)
            name, arg1, arg2 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            # convert relations to camel case
            name = self._normalize_relation_name(name)
            arg2 = arg2.split(",")[0] # strip any <rel-type> defs
            arg1 = arg1.split("|")
            arg2 = arg2.split("|")
            tmp.append((name,arg1,arg2))
        config['relations'] = tmp

        tmp = []
        for item in config['attributes']:
            name, arg = item.split()
            arg = arg.split(":")[-1]
            tmp.append((name, arg))
        config['attributes'] = tmp

        return config


def get_doc_ids_by_query(session, candidate_class, cid_query):
    """
    Given a Candidate.id set  query, return all corresponding parent document ids

    :param session:
    :param candidate_class:
    :param cid_subquery:
    :return:
    """
    arg = [arg + "_id" for arg in candidate_class.__argnames__][0]

    # build set of Span object ids
    q1 = session.query(candidate_class.__dict__[arg]).filter(Candidate.id.in_(cid_query)).subquery()
    q2 = session.query(Span.sentence_id).filter(Span.id.in_(q1)).subquery()
    return session.query(Sentence.document_id).filter(Sentence.id.in_(q2)).distinct()

def get_doc_ids_by_split(session, candidate_class, split):
    """
    Given a Candidate.id set split, return all corresponding parent document ids

    :param session:
    :param candidate_class:
    :param split:
    :return:
    """
    cid_query = session.query(candidate_class.id).filter(candidate_class.split == split)
    return get_doc_ids_by_query(session, candidate_class, cid_query).all()

def get_docs_by_split(session, candidate_class, split):
    """

    :param session:
    :param candidate_class:
    :param split:
    :return:
    """
    cid_subquery = session.query(candidate_class.id).filter(candidate_class.split == 0)
    doc_subquery = get_doc_ids_by_query(session, candidate_class, cid_subquery)
    return session.query(Document).filter(Document.id.in_(doc_subquery)).all()

def get_span_ids_by_cand_query(session, candidate_class, cid_query):
    """
    Return all span_ids for a given candidate set query
    :param session:
    :param candidate_class:
    :param cid_query:
    :return:
    """
    args = [arg + "_id" for arg in candidate_class.__argnames__]
    q1 = session.query(candidate_class.__dict__[args[0]], candidate_class.__dict__[args[1]])
    span_pairs = q1.filter(Candidate.id.in_(cid_query)).all()
    return set(itertools.chain.from_iterable(span_pairs))

def doc_to_text(doc):
    """
    Convert document object to original text represention.
    Assumes parser offsets map to original document offsets

    :param doc:
    :param sent_delim:
    :return:
    """
    text = u""
    for i,sent in enumerate(doc.sentences):
        # setup padding so that BRAT displays a minimal amount of newlines
        # while still preserving char offsets
        if len(text) != sent.abs_char_offsets[0]:
            padding = (sent.abs_char_offsets[0] - len(text))
            text += ' ' * (padding - 1) + u"\n"
        text += sent.text.rstrip(u' \t\n\r')
    return text