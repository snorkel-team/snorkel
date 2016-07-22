import re
from .base import *
from spacy.en import English

#python -m spacy.en.download

class SpacyParser(object):
    '''https://spacy.io/#example-use'''
    def __init__(self, num_threads=4):
        
        self.nlp = English(tokenizer=True, parser=True, tagger=True,
                           entity=None, matcher=None)
    
    def parse(self, doc, doc_id=None):
        """Parse a raw document as a string into a list of sentences"""
        if len(doc.strip()) == 0:
            return
        doc = doc.decode("utf-8")
        for doc in self.nlp.pipe([doc], batch_size=50, n_threads=4):
            assert doc.is_parsed
                    
        for sent_id, sent in enumerate(doc.sents):
            tokens = [t for t in sent]
            token_idxs = [t.idx for t in sent]
            words = [t.text for t in sent]
            lemmas = [self.nlp.vocab.strings[t.lemma] for t in tokens]
            poses = [self.nlp.vocab.strings[t.tag] for t in tokens]
            dep_labels = [self.nlp.vocab.strings[t.dep] for t in tokens]
            # index tokens to determine sentence offset for dependency tree
            token_idx = {t:i for i,t in enumerate(tokens)}
            dep_parents = [token_idx[t.head] for t in tokens] 
            
            s = Sentence(words=words,lemmas=lemmas,poses=poses, 
                         dep_parents=dep_parents, dep_labels=dep_labels, 
                         sent_id=sent_id, doc_id=doc_id, text=sent.text,
                         token_idxs=token_idxs, doc_name=doc_id )

            yield s
