from collections import namedtuple

Sentence = namedtuple('Sentence', ['words', 'lemmas', 'poses', 'dep_parents',
                                   'dep_labels', 'sent_id', 'doc_id', 'text',
                                   'token_idxs', 'doc_name'])
