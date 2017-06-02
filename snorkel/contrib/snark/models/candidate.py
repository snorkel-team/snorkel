from . import SparkModel

class Candidate(SparkModel):
    """An abstract candidate relation."""
    def __init__(self, id, context_names, contexts, name='Candidate'):
        self.id = id
        self.name = name
        self.__argnames__ = context_names
        for name, context in zip(context_names, contexts):
            setattr(self, name, context)

    def get_contexts(self):
        """Get a tuple of the consituent contexts making up this candidate"""
        return tuple(getattr(self, name) for name in self.__argnames__)

    def get_parent(self):
        # Fails if both contexts don't have same parent
        p = [c.get_parent() for c in self.get_contexts()]
        if p.count(p[0]) == len(p):
            return p[0]
        else:
            raise Exception("Contexts do not all have same parent")

    def get_cids(self):
        """Get a tuple of the canonical IDs (CIDs) of the contexts making up 
        this candidate.
        """
        return tuple(getattr(self, name + "_cid") for name in self.__argnames__)

    def __len__(self):
        return len(self.__argnames__)

    def __getitem__(self, key):
        return self.get_contexts()[key]

    def __repr__(self):
        return "%s(%s)" % (self.name, ", ".join(map(str, self.get_contexts())))
