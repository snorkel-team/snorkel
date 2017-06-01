from . import SparkModel


class Candidate(SparkModel):
    def __init__(self, relation):
        self.relation = relation

    def get_contexts(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.get_contexts())

    def __getitem__(self, key):
        return self.get_contexts()[key]

    def __repr__(self):
        return "%s(%s)" % (self.relation, ", ".join(map(str, self.get_contexts())))


class UnaryCandidate(Candidate):
    def __init__(self, relation, context):
        super(UnaryCandidate, self).__init__(relation)
        self.context = context

    def get_contexts(self):
        return self.context,


class BinaryCandidate(Candidate):
    def __init__(self, relation, context1, context2):
        super(BinaryCandidate, self).__init__(relation)
        self.context1 = context1
        self.context2 = context2

    def get_contexts(self):
        return self.context1, self.context2


class TernaryCandidate(Candidate):
    def __init__(self, relation, context1, context2, context3):
        super(TernaryCandidate, self).__init__(relation)
        self.context1 = context1
        self.context2 = context2
        self.context3 = context3

    def get_contexts(self):
        return self.context1, self.context2, self.context3
