from .models import CandidateSet, AnnotationKey, AnnotationKeySet, Label


def create_or_fetch(session, set_class, instance_or_name):
    """Returns a named set ORM object given an instance or name as string"""
    if isinstance(instance_or_name, str):
        x = session.query(set_class).filter(set_class.name == instance_or_name).first()
        if x is not None:
            return x
        else:
            x = set_class(name=instance_or_name)
            session.add(x)
            session.commit()
            return x
    elif isinstance(instance_or_name, set_class):
        return instance_or_name
    else:
        return ValueError("%s-type required" % set_class.__class__)


class ExternalAnnotationsLoader(object):
    """Class to load external annotations."""
    def __init__(self, session, candidate_class, candidate_set, annotation_key, expand_candidate_set=False):
        self.session         = session
        self.candidate_class = candidate_class
        self.candidate_set   = create_or_fetch(self.session, CandidateSet, candidate_set)
        self.annotation_key  = create_or_fetch(self.session, AnnotationKey, annotation_key)
        self.expand_candidate_set = expand_candidate_set

        # Create a key set with the same name as the annotation name
        self.key_set = create_or_fetch(self.session, AnnotationKeySet, annotation_key)

        if self.annotation_key not in self.key_set.keys:
            self.key_set.append(self.annotation_key)
        self.session.commit()

    def add(self, temp_contexts):
        """
        Adds a candidate to a new or existing candidate_set.
        
        :param temp_contexts: This is a *dictionary* of *TemporaryContext* objects corresponding to the args of
        the Candidate class.
        """
        # Create Contexts, make sure inserted into DB, and form dict of context ids
        d = {}
        for argname, temp_context in temp_contexts.iteritems():
            temp_context.load_id_or_insert(self.session)
            d[argname + '_id'] = temp_context.id

        # Create Candidate, make sure inserted in DB, and add to candidate set
        # Note: This operation is not performance-oriented (as it is in e.g. CandidateExtractor)
        q = self.session.query(self.candidate_class)
        for k, v in d.iteritems():
            q = q.filter(getattr(self.candidate_class, k) == v)
        candidate = q.first()
        if candidate is None and self.expand_candidate_set:
            candidate = self.candidate_class(**d)
            self.session.add(candidate)
            self.candidate_set.append(candidate)
        elif candidate is None:
            raise ValueError('Candidate %s not found in CandidateSet, and expand_candidate_set is False.'
                             % '-'.join(str(context_id) for context_id in d.values()))

        # Add annotation
        label = Label(key=self.annotation_key, candidate=candidate, value=1)
        self.session.add(label)

        # Commit session
        self.session.commit()
