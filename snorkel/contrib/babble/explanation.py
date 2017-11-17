import re

class Explanation(object):
    def __init__(self, condition, label, candidate=None, name=None, 
                 semantics=None, paraphrase=None):
        """
        Constructs an Explanation object.

        :param condition: A string explanation that expresses a Boolean 
            condition (e.g., "The sentence is at least 5 words long.")
        :param label: The Boolean label to apply to candidates for which the 
            condition evaluates to True.
        :param candidate: A candidate that the explanation is consistent with.
            May be a candidate object or the candidate's stable_id (for linking 
            later.)
        :param name: The name of this explanation.
        :param semantics: The intended semantic representation of the 
            explanation (if known).
        :param paraphrase: A paraphrase of the explanation.
        """
        assert(isinstance(condition, basestring))
        condition = condition.decode('utf-8')
        condition = re.sub('\s+', ' ', condition, flags=re.UNICODE)
        self.condition = condition
        assert(isinstance(label, bool))
        self.label = label
        self.candidate = candidate
        self.name = name
        self.semantics = semantics
        self.paraphrase = paraphrase

    def __hash__(self):
        return hash((self.label, self.condition))

    def __repr__(self):
        if self.name:
            return 'Explanation("%s: %s, %s")' % (self.name, self.label, self.condition.encode('utf-8'))
        else:
            return 'Explanation("%s, %s")' % (self.label, self.condition.encode('utf-8'))
    
    def display(self):
        print 'Explanation'
        print('%-12s %s' % ('condition', self.condition))
        print('%-12s %d' % ('label', self.label))
        print('%-12s %s' % ('candidate', self.candidate))
        print('%-12s %s' % ('name', self.name))
        print('%-12s %s' % ('semantics', self.semantics))