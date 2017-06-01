from snorkel.models.annotation import Label, LabelKey
from snorkel.models.meta import snorkel_conn_string
from snorkel.models.views import create_serialized_candidate_view


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
        create_serialized_candidate_view(snorkel_session, candidate_class, verbose=False)

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

        for lf in LFs:
            # TODO: Create annotation key
            lf_id = None

            labels = self.split_cache[split].map(lambda c: (c.id, lf(c)))
            labels.filter(lambda l: l[1] != 0 and l[1] is not None)
            labels.collect().foreach(lambda y: self._insert_label((y[0], lf_id, y[1])))

        # TODO: Return label matrix

    def _clear_labels(self):
        """
        Clears the database of labels
        """
        self.snorkel_session.session.query(Label).delete(synchronize_session='fetch')

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

        rdd = jdbcDF.rdd.map(wrap_candidate)
        rdd = rdd.setName("Snorkel Candidates, Split " + split + " (" + self.candidate_class.__name__ + ")")
        self.split_cache[split] = rdd.cache()

    def _insert_label(self, candidate_id, lf_id, value):
        """
        Writes label record back to database connected to by snorkel_session.

        :param candidate_id: internal id of the candidate
        :param lf_id: internal id of the labeling function (LabelKey.id)
        :param value: the label value
        """
        # TODO
        pass


def wrap_candidate(row):
    """
    Wraps raw tuple from <candidate_classname>_serialized table with object data structure

    :param row: raw tuple
    :return: candidate object
    """
    # TODO
    return row
