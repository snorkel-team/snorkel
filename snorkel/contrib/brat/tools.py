import os
import re
import sys
import glob
import codecs
from sqlalchemy.sql import select
from collections import defaultdict
from ...db_helpers import reload_annotator_labels
from ...parser import TextDocPreprocessor, CorpusParser
from ...models import Candidate, StableLabel, Document, TemporarySpan, Sentence, candidate_subclass


class Brat(object):
    """
    Snorkel Import/Export for
    Brat Rapid Annotation Tool
    http://brat.nlplab.org/

    Brat uses standoff annotation format (see: http://brat.nlplab.org/standoff.html)

    Annotation ID Types
    T: text-bound annotation
    R: relation
    E: event
    A: attribute
    M: modification (alias for attribute, for backward compatibility)
    N: normalization [new in v1.3]
    #: note

    Caveats:
    (1) Attributes, normalization, and notes are added as candidate meta information

    """

    TEXT_BOUND_ID = 'T'
    RELATION_ID = 'R'
    EVENT_ID = 'E'
    ATTRIB_ID = 'A'
    MOD_ID = 'M'
    NORM_ID = 'N'
    NOTE_ID = '#'

    def __init__(self, session, tmpl_path='tmpl.config', encoding="utf-8", verbose=True):
        """

        :param session:
        :param tmpl_path:
        :param encoding:
        :param verbose:
        """
        self.session = session
        self.encoding = encoding
        self.verbose = verbose

        # load brat config template
        mod_path = "{}/{}".format(os.path.abspath(os.path.dirname(__file__)), tmpl_path)
        self.brat_tmpl = "".join(open(mod_path, "rU").readlines())

        # snorkel dynamic types
        self.subclasses = {}

    def import_project(self, input_dir, annotations_only=True, annotator_name='brat', num_threads=1, parser=None):
        """

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
        config = self._load_config(config_path)
        anno_filelist = set([os.path.basename(fn).strip(".ann") for fn in glob.glob(input_dir + "/*.ann")])

        # import standoff annotations for all documents
        annotations = {}
        for fn in anno_filelist:
            txt_fn = "{}/{}.txt".format(input_dir, fn)
            ann_fn = "{}/{}.ann".format(input_dir, fn)
            if os.path.exists(txt_fn) and os.path.exists(ann_fn):
                annotations[fn] = self._parse_annotations(txt_fn, ann_fn)

        # by default, we parse and import all project documents
        if not annotations_only:
            self._parse_documents(input_dir + "/*.txt", num_threads, parser)

        # create types
        self._create_candidate_subclasses(config)

        # create candidates
        self._create_candidates(annotations, annotator_name)


    def export_project(self, output_dir, positive_only_labels=True):
        """

        :param output_dir:
        :return:
        """
        candidates = self.session.query(Candidate).filter(Candidate.split == 0).all()
        doc_index = _group_by_document(candidates)
        snorkel_types = {type(c): 1 for c in candidates}

        for name in doc_index:
            print name
            for c in doc_index[name]:
                print c
                text = "".join([s.text for s in c[0].sentence.document.sentences])
                print text

            fp = "{}/{}".format(output_dir,name)

            break

    def _parse_documents(self, input_path, num_threads, parser):
        """

        :param input_path:
        :param num_threads:
        :param parser:
        :return:
        """
        doc_preprocessor = TextDocPreprocessor(path=input_path, encoding=self.encoding)
        corpus_parser = CorpusParser(parser)
        corpus_parser.apply(doc_preprocessor, parallelism=num_threads)

    def _parse_annotations(self, txt_filename, ann_filename):
        """
        Use parser to import BRAT backoff format
        TODO: Currently only supports Entities & Relations

        :param txt_filename:
        :param ann_filename:
        :return:
        """
        annotations = {}

        # load document
        doc = []
        with codecs.open(txt_filename, "rU", encoding=self.encoding) as fp:
            for line in fp:
                doc += [line.strip().split()]

        # build doc string and char to word index
        doc_str = ""
        char_idx = {}
        for i, sent in enumerate(doc):
            for j in range(0, len(sent)):
                char_idx[len(doc_str)] = (i, j)
                for ch in sent[j]:
                    doc_str += ch
                    char_idx[len(doc_str)] = (i, j)
                doc_str += " " if j != len(sent) - 1 else "\n"
        doc_str = doc_str.strip()

        # load annotations
        with codecs.open(ann_filename, "rU", encoding=self.encoding) as fp:
            for line in fp:
                row = line.strip().split("\t")
                anno_id_prefix = row[0][0]

                if anno_id_prefix == Brat.TEXT_BOUND_ID:
                    anno_id, entity, text = row
                    entity_type = entity.split()[0]
                    spans = map(lambda x: map(int, x.split()),
                                entity.lstrip(entity_type).split(";"))

                    # discontinuous mentions
                    if len(spans) != 1:
                        print>> sys.stderr, "NotImplementedError: Discontinuous Spans"
                        continue

                    entity = []
                    for (i, j) in spans:
                        if i in char_idx:
                            mention = doc_str[i:j]
                            tokens = mention.split()
                            sent_id, word_offset = char_idx[i]
                            word_mention = doc[sent_id][word_offset:word_offset + len(tokens)]
                            parts = {"sent_id":sent_id,"char_start":i,"char_end":j, "entity_type":entity_type,
                                     "idx_span":(word_offset, word_offset + len(tokens)), "span":word_mention}
                            entity += [parts]
                        else:
                            print>> sys.stderr, "SUB SPAN ERROR", text, (i, j)
                            continue

                    # TODO: we assume continuous spans here
                    annotations[anno_id] = entity if not entity else entity[0]

                elif anno_id_prefix == Brat.RELATION_ID:
                    anno_id, rela = row
                    rela_type, arg1, arg2 = rela.split()

                    arg1 = arg1.split(":")[1] if ":" in arg1 else arg1
                    arg2 = arg2.split(":")[1] if ":" in arg2 else arg2

                    annotations[anno_id] = (rela_type, arg1, arg2)

                elif anno_id_prefix == Brat.EVENT_ID:
                    print>> sys.stderr, "NotImplementedError: Events"
                    raise NotImplementedError

                elif anno_id_prefix == Brat.ATTRIB_ID:
                    print>> sys.stderr, "NotImplementedError: Attributes"

        return annotations

    def _create_candidate_subclasses(self, config):
        """
        Given a BRAT config file, create Snorkel candidate subclasses.

        :param config:
        :return:
        """
        for class_name in config['entities']:
            try:
                self.subclasses[class_name] = candidate_subclass(class_name, [class_name.lower()])
                print 'Entity({},[{}])'.format(class_name, class_name.lower())
            except:
                pass

        for item in config['relations']:
            m = re.search("^(.+)\s+Arg1:(.+),\s*Arg2:(.+),*\s*(.+)*$", item)
            name, arg1, arg2 = m.group(1), m.group(2), m.group(3)

            arg1 = arg1.lower().split("|")
            arg2 = arg2.lower().split("|")

            # TODO: Assume simple relation types *without* multiple argument types
            if (len(arg1) > 1 or len(arg2) > 1) and arg1 != arg2:
                print>>sys.stderr,"Error: Snorkel does not support multiple argument types"

            try:
                args = sorted(set(arg1 + arg2))
                self.subclasses[name] = candidate_subclass(name, args)
                print 'Relation({},{})'.format(name, args)
            except:
                pass

    def _load_config(self, filename):
        """

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

        return config

    def _create_config(self, candidate_type):
        """
        Export a minimal BRAT configuration schema defining
        a binary relation and two argument types.

        TODO: Model richer semantics here (asymmetry, n-arity relations)

        :param candidate_type:
        :return:
        """
        rel_type = str(candidate_type.type).rstrip(".type")
        arg_types = [key.rstrip("_cid") for key in candidate_type.__dict__ if "_cid" in key]

        entity_defs = "\n".join(arg_types)
        rela_def = "Arg1:{}, Arg2:{}".format(*arg_types) if len(arg_types) == 2 else ""
        return self.brat_tmpl.format(entity_defs, rela_def, "", "")


    def _create_candidates(self, annotations, annotator_name, clear=True):
        """

        :return:
        """
        # create stable annotation labels
        stable_labels_by_type = defaultdict(list)

        for name in annotations:
            if annotations[name]:
                spans = [key for key in annotations[name] if key[0] == Brat.TEXT_BOUND_ID]
                relations = [key for key in annotations[name] if key[0] == Brat.RELATION_ID]

                # create span labels
                spans = {key:"{}::span:{}:{}".format(name, annotations[name][key]["char_start"],
                                                     annotations[name][key]["char_end"]) for key in spans}
                for key in spans:
                    entity_type = annotations[name][key]['entity_type']
                    stable_labels_by_type[entity_type].append(spans[key])

                # create relation labels
                for key in relations:
                    rela_type, arg1, arg2 = annotations[name][key]
                    rela = sorted([[annotations[name][arg1]["entity_type"], spans[arg1]],
                                    [annotations[name][arg2]["entity_type"],spans[arg2]]])
                    stable_labels_by_type[rela_type].append("~~".join(zip(*rela)[1]))

        # create stable labels
        # NOTE: we store each label class type in a different split so that it is compatible with
        # the current version of 'reload_annotator_labels', where we create candidates by split id
        for i, class_type in enumerate(stable_labels_by_type):

            for context_stable_id in stable_labels_by_type[class_type]:
                query = self.session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_id)
                query = query.filter(StableLabel.annotator_name == annotator_name)
                if query.count() != 0:
                    continue
                self.session.add(StableLabel(context_stable_ids=context_stable_id, split=i,
                                             annotator_name=annotator_name, value=1))

        abs_offsets = {}
        entity_types = defaultdict(list)
        for i, class_type in enumerate(stable_labels_by_type):

            class_name = self.subclasses[class_type]
            for et in stable_labels_by_type[class_type]:
                contexts = et.split('~~')
                spans = []
                for c,et in zip(contexts,class_name.__argnames__):
                    stable_id = c.split(":")
                    name, offsets = stable_id[0], stable_id[-2:]
                    span = map(int, offsets)
                    if name not in abs_offsets:
                        doc = self.session.query(Document).filter(Document.name==name).one()
                        abs_offsets[name] = abs_doc_offsets(doc)

                    for j,offset in enumerate(abs_offsets[name]):
                        if span[0] >= offset[0] and span[1] <= offset[1]:
                            tc = TemporarySpan(char_start=span[0]-offset[0], char_end=span[1]-offset[0]-1,
                                               sentence=doc.sentences[j])
                            tc.load_id_or_insert(self.session)
                            spans.append(tc)

                entity_types[class_type].append(spans)

        for i, class_type in enumerate(stable_labels_by_type):

            if clear:
                self.session.query(Candidate).filter(Candidate.split == i).delete()

            candidate_args = {'split': i}
            for args in entity_types[class_type]:
                for j, arg_name in enumerate(self.subclasses[class_type].__argnames__):
                    candidate_args[arg_name + '_id'] = args[j].id

                candidate = self.subclasses[class_type](**candidate_args)
                self.session.add(candidate)

        self.session.commit()


def _group_by_document(candidates):
    """

    :param candidates:
    :return:
    """
    doc_index = defaultdict(list)
    for c in candidates:
        name = c[0].sentence.document.name
        doc_index[name].append(c)
    return doc_index


def abs_doc_offsets(doc):
    """

    :param doc:
    :return:
    """
    abs_char_offsets = []
    for sent in doc.sentences:
        stable_id = sent.stable_id.split(":")
        name, offsets = stable_id[0], stable_id[-2:]
        offsets = map(int, offsets)
        abs_char_offsets.append(offsets)
    return abs_char_offsets