from snorkel.models.annotation import Label, LabelKey
from snorkel.models.meta import snorkel_conn_string
from snorkel.models.views import create_serialized_candidate_view


class SparkLabelAnnotator:
    def __init__(self, snorkel_session, spark_session, candidate_class):
        self.snorkel_session = snorkel_session
        self.spark_session = spark_session
        self.candidate_class = candidate_class

        self.split_cache = {}

        # TODO: Either check for existence or change method to be forgiving
        create_serialized_candidate_view(snorkel_session, candidate_class, verbose=False)

    def apply(self, LFs, split):
        self.clear_labels(split)

        if split not in self.split_cache:
            self.load_candidates(split)



    def clear_labels(self, split):
        self.snorkel_session.session.query(Label).delete(synchronize_session='fetch')

        query = self.snorkel_session.query(LabelKey).filter(LabelKey.group == 0)
        query.delete(synchronize_session='fetch')

    def load_candidates(self, split):
        jdbcDF = self.spark_session.read \
            .format("jdbc") \
            .option("url", "jdbc:" + snorkel_conn_string) \
            .option("dbtable", self.candidate_class.__tablename__ + "_serialized") \
            .load()

        rdd = jdbcDF.rdd.setName("Snorkel Candidates, Split " + split + " (" + self.candidate_class.__name__ + ")")
        self.split_cache[split] = rdd.cache()
