from . import SparkModel

class Candidate(SparkModel):
    """An abstract candidate relation."""
    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

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
        return "%s(%s)" % (self.__class__.__name__, ", ".join(
            map(str, self.get_contexts())))


def candidate_subclass(class_name, args):
    """
    Creates and returns a Candidate subclass with provided argument names, which
    are Context type.
    Creates the table in DB if does not exist yet.

    :param class_name: The name of the class, should be "camel case" e.g. 
        NewCandidateClass
    :param args: A list of names of consituent arguments, which refer to the 
        Contexts--representing mentions--that comprise the candidate.
    """
    class_attribs = {
        'id': None,

        # Helper method to get argument names
        '__argnames__' : args
    }
        
    # Create named arguments
    for arg in args:
        class_attribs[arg + '_id'] = None
        class_attribs[arg] = None
        class_attribs[arg + '_cid'] = None

    # Create class
    return type(class_name, (Candidate,), class_attribs)
