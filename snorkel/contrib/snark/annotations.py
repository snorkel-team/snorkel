from snorkel.annotations import load_label_matrix
from snorkel.models.annotation import Label, LabelKey
from snorkel.models.meta import snorkel_conn_string
from snorkel.models.views import create_serialized_candidate_view
from models.candidate import candidate_subclass
from models.context import Sentence, Span
from functools import partial


class SparkLabelAnnotator:
    """
    Distributes candidates to a Spark cluster and applies labeling functions over them.

    Distributed candidate sets are cached by split for the annotator, so repeated calls to apply() will
    operate over the same split of candidates as the first call for that split, regardless of how the
    underlying database changes. To reload a candidate set split, construct a new SparkLabelAnnotator.
    """
    def __init__(self, snorkel_session, spark_session, candidate_class):
        """
        Constructor

        :param snorkel_session: the SnorkelSession for the Snorkel application
        :param spark_session: a PySpark SparkSession
        :param candidate_class: the subclass of Candidate to be labeled
        """
        self.snorkel_session = snorkel_session
        self.spark_session = spark_session
        self.candidate_class = candidate_class

        self.split_cache = {}

        # TODO: Either check for existence or change method to be forgiving
        #create_serialized_candidate_view(snorkel_session, candidate_class, verbose=False)

    def apply(self, LFs, split):
        """
        Applies a collection of labeling functions to a split of candidates.

        The results are persisted in the Snorkel database.

        :param LFs: collection of labeling functions
        :param split: the split of candidates to label
        :return: a csr_LabelMatrix of the resulting labels
        """
        self._clear_labels()

        if split not in self.split_cache:
            self._load_candidates(split)

        key_query = LabelKey.__table__.insert()
        label_query = Label.__table__.insert()

        for lf in LFs:
            lf_id = self.snorkel_session.execute(key_query, {'name': lf.__name__, 'group': 0}).inserted_primary_key[0]

            labels = self.split_cache[split].map(lambda c: (c.id, lf(c)))
            labels.filter(lambda (_, value): value != 0 and value is not None)
            for cid, value in labels.toLocalIterator():
                self.snorkel_session.execute(label_query, {'candidate_id': cid, 'key_id': lf_id, 'value': value})

        return load_label_matrix(self.snorkel_session, split=split)

    def _clear_labels(self):
        """
        Clears the database of labels
        """
        self.snorkel_session.query(Label).delete(synchronize_session='fetch')

        query = self.snorkel_session.query(LabelKey).filter(LabelKey.group == 0)
        query.delete(synchronize_session='fetch')

    def _load_candidates(self, split):
        """
        Loads a set of candidates as a Spark RDD and caches the results as self.split_cache[split]

        :param split: the split of candidates to load
        """
        jdbcDF = self.spark_session.read \
            .format("jdbc") \
            .option("url", "jdbc:" + snorkel_conn_string) \
            .option("dbtable", self.candidate_class.__tablename__ + "_serialized") \
            .load()

        rdd = jdbcDF.rdd.map(partial(
            wrap_candidate,
            class_name=self.candidate_class.__name__,
            argnames=self.candidate_class.__argnames__
        ))
        rdd = rdd.setName("Snorkel Candidates, Split " + str(split) + " (" + self.candidate_class.__name__ + ")")
        self.split_cache[split] = rdd.cache()


# Note: This should ideally not be hard-coded...
CONTEXT_OFFSET = 2
# Note: We store the sentence here, not the sentence_id
SPAN_COLS = ['id', 'sentence', 'char_start', 'char_end', 'meta']
SENTENCE_COLS = ['id', 'document_id', 'position', 'text', 'words', 
    'char_offsets', 'lemmas', 'pos_tags', 'ner_tags', 'dep_parents',
    'dep_labels', 'entity_cids', 'entity_types']

def wrap_candidate(row, class_name='Candidate', argnames=[]):
    """
    Wraps raw tuple from <candidate_classname>_serialized table with object data
    structure

    :param row: raw tuple
    :return: candidate object
    """
    # Infer arity from size of row
    arity = float(len(row) - 2 - len(SENTENCE_COLS)) / len(SPAN_COLS)
    assert int(arity) == arity
    arity = int(arity)

    # NB: We hardcode in an assumed Context hierarchy here:
    # Sentence -> (Spans)
    # Should make more general. Also note that we assume only the local context
    # subtree (Sentence + k Spans comprising the candidate) are provided.

    # Create Sentence object
    sent = Sentence(**dict(zip(SENTENCE_COLS, row[-len(SENTENCE_COLS):])))

    # Create the Span objects
    spans = []
    for i in range(arity):
        j = CONTEXT_OFFSET + i * len(SPAN_COLS)
        span = Span(**dict(zip(SPAN_COLS, row[j:j + len(SPAN_COLS)])))
        # Store the Sentence in the Span
        span.sentence = sent
        spans.append(span)

    # Create candidate object
    Candidate = candidate_subclass(class_name, argnames)
    attrib_names = ['id'] + argnames
    return Candidate(**dict(zip(attrib_names, [row[0]] + spans)))
