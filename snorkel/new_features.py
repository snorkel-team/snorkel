import os, sys
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
import itertools
from pandas import DataFrame
import re
# Feature modules
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from utils import get_as_dict
from entity_features import *

import math
import numpy as np
from multiprocessing import Process, Queue

import string
import fuzzy
import pyphen
#pyphen.language_fallback('nl_NL_variant1')
morphology = pyphen.Pyphen(lang="en_Latn_US")
#morphology = pyphen.Pyphen(lang="en_US")
morphology = pyphen.Pyphen(lang="nl_NL_variant1")

soundex = fuzzy.Soundex(4)


def letter_ratio(c,idxs,bins=20):
    s = c.get_attrib_span("words")
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
    c = float(sum([1 for ch in s if ch in punc]))
    w = (1.0 - (c/float(len(s)))) * 100
    w = int(w)
    return "LETTER_RATIO_[{}]".format( int(w/float(bins)) )
    
    
def vowel_ratio(c,idxs,bins=20):
    s = c.get_attrib_span("words")
    punc = 'aeiou'
    c = float(sum([1 for ch in s if ch in punc]))
    w = ((c/float(len(s)))) * 100
    w = int(w)
    #return int(w/float(bins))
    return "VOWEL_RATIO_[{}]".format( int(w/float(bins)) )


def word_soundex(c,idxs):
    s = c.get_attrib_span("words")
    tokens = s.split()
    for i in range(0,len(tokens)):
        seq = map(soundex,tokens[0:i+1])
        yield "SOUNDEX_SEQ_[{}]".format(" ".join(seq))


def affex_norm(affex):
    affex = affex.lower() 
    if affex.isdigit():
        affex = "D"
    elif affex in string.punctuation:
        affex = "P"
    return affex


def affexes(c,idxs):
    s = c.get_attrib_span("words")
    tokens = s.split()
    
    seq = morphology.inserted(tokens[0])
    t = seq.split("-")
    yield "PREFIX_FW_[{}]".format(affex_norm(t[0]))
    if len(t) > 1:
        yield "SUFFIX_FW_[{}]".format(affex_norm(t[-1]))
          
    if len(tokens) > 1:
        seq = morphology.inserted(tokens[-1])
        t = seq.split("-")
        yield "PREFIX_LW_[{}]".format(affex_norm(t[0]))
        if len(t) > 1:
            yield "SUFFIX_LW_[{}]".format(affex_norm(t[-1]))
        
        
def affexes2(c,idxs):
    s = c.get_attrib_span("words")
    ftr = morphology.inserted(s)
    t = ftr.split("-")
    yield "PREFIX_[{}]".format(t[0].lower())
    if len(t) > 1:
        yield "SUFFIX_[{}]".format(t[-1].lower())


def word_seq_affixes(c,idxs):
    s = c.get_attrib_span("words")
    tokens = s.split()
    for t in tokens:
        m = morphology.inserted(t).split("-")
        if len(m) == 1:
            yield u"MORPHEME_FREE_[{}]".format(affex_norm(m[0]))
        else:
            yield u"MORPHEME_PREFIX_[{}]".format(affex_norm(m[0]))
            yield u"MORPHEME_SUFFIX_[{}]".format(affex_norm(m[-1]))
                

def word_shape_seq(c,idxs):
    words = c.get_attrib_span("words")
    yield "[{}]".format(word_shape(words)) 
    tokens = words.split()
    if len(tokens) > 1:
        for w in tokens:
            yield "SEQ_[{}]".format(word_shape(w))
    

def word_shape(s):
    '''From SpaCY'''
    if len(s) >= 100:
        return 'LONG'
    length = len(s)
    shape = []
    last = ""
    shape_char = ""
    seq = 0
    for c in s:
        if c.isalpha():
            if c.isupper():
                shape_char = "X"
            else:
                shape_char = "x"
        elif c.isdigit():
            shape_char = "d"
        else:
            shape_char = c
        if shape_char == last:
            seq += 1
        else:
            seq = 0
            last = shape_char
        if seq < 4:
            shape.append(shape_char)
    
    return ''.join(shape)
   

def left_window(m, window=3, match_attrib="lemmas"):
    idx = max(0,min(m.idxs) - window)
    span = range(idx,min(m.idxs))
    return [m.get_attrib(match_attrib)[i] for i in span]


def right_window(m, window=3, match_attrib="lemmas"):
    idx = min(len(m.get_attrib()), max(m.idxs) + window + 1)
    span = range(max(m.idxs) + 1,idx)
    return [m.get_attrib(match_attrib)[i] for i in span]

rgx_is_digit = re.compile("([0-9]+[,.]*)+")

def word_seq(c,idxs):
    '''Linear chain within mention'''
    words = c.get_attrib_tokens("lemmas")
    lw = left_window(c,window=1)
    rw = right_window(c,window=1)
    lw = u"_" if not lw else lw[0]
    rw = u"_" if not rw else rw[0]
    lw = u"NUMBER" if rgx_is_digit.search(lw) else lw
    rw = u"NUMBER" if rgx_is_digit.search(rw) else rw
    
    for i in range(len(words)):
        left = rw if i == 0 else words[i-1]
        right = lw if i == len(words) - 1 else words[i+1]
        yield u"W_LEMMA_L_[{}]".format(left)
        yield u"W_LEMMA_R_[{}]".format(right)
   
   
def morpheme_seq(c,idxs,ngr=2):
    s = c.get_attrib_span("lemmas")
    tokens = s.split()
    
    tmpl = u"MORPHEME_SEQ_[{}]"
    for i in range(0,len(tokens)):  
        seq = morphology.inserted(tokens[i]).split("-")
        seq = map(lambda x:re.sub("\d",u"D",x.lower()),seq)
        seq = map(lambda x:re.sub("[.()\]\['-]",u"P",x.lower()),seq)
        if len(seq) > 1:
            for j in range(0,len(seq)-ngr+1):
                v = u"" if j == 0 else u"-"
                v += u"".join(seq[j:j+2])
                v += u"" if j+2 == len(seq) else u"-"
                yield tmpl.format(v)

'''
WORD_L_[acid]
WORD_R_[alpha]

MORPHEME_SEQ_[an-ti-]
MORPHEME_SEQ_[-al-ly]
MORPHEME_SEQ_[-in-flamm-]

WORD_SHAPE_[DXX]
WORD_SHAPE_SEQ_[XX-d]

WORD_PREFIX_[an]
WORD_SUFFIX_[ly]

SOUNDEX_[AD32]

ALL_UPPERCASE
PARANTHETICAL
LEFT_OF_PARANTHETICAL
'''
#PARANTHETICAL_DEFINED




def generate_mention_feats(get_feats, prefix, candidates):
    for i,c in enumerate(candidates):
        for ftr in get_feats(c):
            yield i, prefix + ftr


class FeaturizerMP(object):
    
    def __init__(self, num_procs=1):
        self.num_procs      = num_procs
        self.feat_index     = None
        self.feat_inv_index = None
    
    @staticmethod
    def featurizer_worker(pid,idxs,candidates,queue): 
        print "\tFeaturizer process_id={} {} items".format(pid, len(idxs))
        block = [candidates[i] for i in idxs]
        feature_generators = FeaturizerMP.apply(block)
        ftr_index = defaultdict(list)
        for i,ftr in itertools.chain(*feature_generators):
            ftr_index[ftr].append(idxs[i])
            
        outdict = {pid:ftr_index}
        queue.put(outdict)
    
    @staticmethod
    def generate_feats(get_feats, prefix, candidates):
        for i,c in enumerate(candidates):
            for f in get_feats(c):
                yield i, prefix + f
   
    @staticmethod
    def preprocess(candidates):
        for c in candidates:
            if not isinstance(c.sentence, dict):
                c.sentence = get_as_dict(c.sentence)
            if c.sentence['xmltree'] is None:
                c.sentence['xmltree'] = corenlp_to_xmltree(c.sentence)
        return candidates
    
    @staticmethod
    def get_features_by_candidate(candidate):
        feature_generators = FeaturizerMP.apply(FeaturizerMP.preprocess([candidate]))
        feats = []
        for i,f in itertools.chain(*feature_generators):
            feats.append(f)
        return feats

    @staticmethod
    def apply(candidates):
        
        feature_generators = []
        
        # Add DDLIB entity features
        feature_generators.append(FeaturizerMP.generate_feats( \
            lambda c : get_ddlib_feats(c, range(c.word_start, c.word_end+1)), 'DDLIB_', candidates))

        # Add TreeDLib entity features
        get_feats = compile_entity_feature_generator()
        feature_generators.append(FeaturizerMP.generate_feats( \
            lambda c : get_feats(c.sentence['xmltree'].root, range(c.word_start, c.word_end+1)), 'TDL_', candidates))
        
        # word shape features
        feature_generators.append( FeaturizerMP.generate_feats( \
            lambda c: word_shape_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
        
        # soundex
        #feature_generators.append( generate_mention_feats( \
        #    lambda c: word_soundex(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
         
        # morphemes
        feature_generators.append( FeaturizerMP.generate_feats( \
            lambda c: morpheme_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
        
        # affixes
        feature_generators.append( FeaturizerMP.generate_feats( \
            lambda c: word_seq_affixes(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
        
        # mention word linear chain
        feature_generators.append( FeaturizerMP.generate_feats( \
            lambda c: word_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
        return feature_generators

    def top_features(self, w, n_max=100):
        """Return a DataFrame of highest (abs)-weighted features"""
        idxs = np.argsort(np.abs(w))[::-1][:n_max]
        d = {'j': idxs, 'w': [w[i] for i in idxs]}
        return DataFrame(data=d, index=[self.feat_inv_index[i] for i in idxs])
    
    def fit(self,candidates):
        
        self.feat_index = {}
        self.feat_inv_index = {}
        candidates = FeaturizerMP.preprocess(candidates)
        
        if self.num_procs > 1:    
            
            out_queue = Queue()
            chunksize = int(math.ceil(len(candidates) / float(self.num_procs)))
            procs = []

            nums = range(0,len(candidates))
            for i in range(self.num_procs):
                p = Process(
                            target=FeaturizerMP.featurizer_worker,
                            args=(i, nums[chunksize * i:chunksize * (i + 1)],
                                  candidates,
                                  out_queue))
                procs.append(p)
                p.start()

            resultdict = {}
            for i in range(self.num_procs):
                r = out_queue.get()
                resultdict.update(r)
             
            for p in procs:
                p.join()
        
            # merge feature    
            f_index = defaultdict(list)
            for i in resultdict: 
                for ftr in resultdict[i]:
                    f_index[ftr] += resultdict[i][ftr]
        
        else:
            feature_generators = FeaturizerMP.apply(candidates)
            f_index = defaultdict(list)
            for i,f in itertools.chain(*feature_generators):
                f_index[f].append(i)
        
        for j,f in enumerate(sorted(f_index.keys())):
            self.feat_index[f] = j
            self.feat_inv_index[j] = f
        
        self.f_index = f_index
        
    def fit_transform(self, candidates):
        self.fit(candidates)
        return self.transform(candidates)
    
    def transform(self,candidates):
        if not self.f_index:
            raise Exception('model is not fit')
        
        F = sparse.lil_matrix((len(candidates), len(self.f_index.keys())))
        for f in sorted(self.f_index.keys()):
            j = self.feat_index[f]
            for i in self.f_index[f]:
                F[i,j] = 1
        return F
        
        


class Featurizer(object):
    """
    A Featurizer applies a set of **feature generators** to each Candidate,
    based on (i) the arity of the candidate, and (ii) the _associated Contexts_.

    The transform() function takes in N candidates, and returns an N x F sparse matrix,
    where F is the dimension of the feature space.
    """
    def __init__(self, arity=1):
        self.arity          = arity
        self.feat_index     = None
        self.feat_inv_index = None

    def _generate_context_feats(self, get_feats, prefix, candidates):
        """
        Given a function that given a candidate, generates features, _using a specific context_,
        and a unique prefix string for this context, return a generator over features (as strings).
        """
        for i,c in enumerate(candidates):
            for f in get_feats(c):
                yield i, prefix + f

    # TODO: Take this out...
    def _preprocess_candidates(self, candidates):
        return candidates

    def _match_contexts(self, candidates):
        """Given the candidates, and using _generate_context_feats, return a list of generators."""
        raise NotImplementedError()

    def transform(self, candidates):
        """Given feature set has already been fit, simply apply to candidates."""
        F                  = sparse.lil_matrix((len(candidates), len(self.feat_index.keys())))
        feature_generators = self._match_contexts(self._preprocess_candidates(candidates))
        
        for i,f in itertools.chain(*feature_generators):
            if self.feat_index.has_key(f):
                F[i,self.feat_index[f]] = 1
        return F

    #Featurizer._match_contexts(self._preprocess_candidates(candidates))


    def fit_transform(self, candidates):
        """Assembles the set of features to be used, and applies this transformation to the candidates"""
        feature_generators = self._match_contexts(self._preprocess_candidates(candidates))

        # Assemble and return the sparse feature matrix
        f_index = defaultdict(list)
        for i,f in itertools.chain(*feature_generators):
            f_index[f].append(i)

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feat_index     = {}
        self.feat_inv_index = {}
        F                   = sparse.lil_matrix((len(candidates), len(f_index.keys())))
        for j,f in enumerate(sorted(f_index.keys())):
            self.feat_index[f] = j
            self.feat_inv_index[j] = f
            for i in f_index[f]:
                F[i,j] = 1
        return F

    def get_features_by_candidate(self, candidate):
        feature_generators = self._match_contexts(self._preprocess_candidates([candidate]))
        feats = []
        for i,f in itertools.chain(*feature_generators):
            feats.append(f)
        return feats

    def top_features(self, w, n_max=100):
        """Return a DataFrame of highest (abs)-weighted features"""
        idxs = np.argsort(np.abs(w))[::-1][:n_max]
        d = {'j': idxs, 'w': [w[i] for i in idxs]}
        return DataFrame(data=d, index=[self.feat_inv_index[i] for i in idxs])



class NgramFeaturizer(Featurizer):
    """Feature for relations (of arity >= 1) defined over Ngram objects."""
    def _preprocess_candidates(self, candidates):
        for c in candidates:
            if not isinstance(c.sentence, dict):
                c.sentence = get_as_dict(c.sentence)
            if c.sentence['xmltree'] is None:
                c.sentence['xmltree'] = corenlp_to_xmltree(c.sentence)
        return candidates

    def _match_contexts(self, candidates):
        feature_generators = []
        
        # Unary relations
        if self.arity == 1:

            # Add DDLIB entity features
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, range(c.word_start, c.word_end+1)), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            get_feats = compile_entity_feature_generator()
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_feats(c.sentence['xmltree'].root, range(c.word_start, c.word_end+1)), 'TDL_', candidates))
            
            # word shape features
            feature_generators.append( generate_mention_feats( \
                lambda c: word_shape_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # ***
            # soundex
            #feature_generators.append( generate_mention_feats( \
            #    lambda c: word_soundex(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
             
            # morphemes
            feature_generators.append( generate_mention_feats( \
                lambda c: morpheme_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # affixes
            feature_generators.append( generate_mention_feats( \
                lambda c: word_seq_affixes(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            # mention word linear chain
            feature_generators.append( generate_mention_feats( \
                lambda c: word_seq(c, range(c.word_start, c.word_end+1)), "WS_", candidates) )
            
            
        if self.arity == 2:
            raise NotImplementedError("Featurizer needs to be implemented for binary relations!")
        return feature_generators


class LegacyCandidateFeaturizer(Featurizer):
    """Temporary class to handle v0.2 Candidate objects."""
    def _match_contexts(self, candidates):
        feature_generators = []

        # Unary relations
        if self.arity == 1:

            # Add DDLIB entity features
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, c.idxs), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            if candidates[0].root is not None:
                get_feats = compile_entity_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.root, c.idxs), 'TDLIB_', candidates))

        if self.arity == 2:

            # Add TreeDLib relation features
            if candidates[0].root is not None:
                get_feats = compile_relation_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.root, c.e1_idxs, c.e2_idxs), 'TDLIB_', candidates))
        return feature_generators

