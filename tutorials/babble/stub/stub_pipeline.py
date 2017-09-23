from snorkel.contrib.babble.pipelines import SnorkelPipeline

class StubPipeline(SnorkelPipeline):
    def __init__(self, session, candidate_class, config):
        self.config = config

    def parse(self):
        print("I parsed!")

    def extract(self):
        print("I extracted!")

    def load_gold(self):
        print("I loaded gold!")

    def collect(self):
        print("I collected signal!")

    def label(self):
        print("I labeled!")

    def supervise(self):
        print("I supervised!")

    def classify(self):
        print("I classified!")