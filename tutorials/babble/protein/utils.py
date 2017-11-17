import bz2
import os
import re
from six.moves.cPickle import load

from string import punctuation
from subprocess import Popen, PIPE, STDOUT


def offsets_to_token(left, right, offset_array, lemmas, punc=set(punctuation)):
    token_start, token_end = None, None
    for i, c in enumerate(offset_array):
        if left >= c:
            token_start = i
        if c > right and token_end is None:
            token_end = i
            break
    token_end = len(offset_array) - 1 if token_end is None else token_end
    token_end = token_end - 1 if lemmas[token_end - 1] in punc else token_end
    return range(token_start, token_end)

class GeniaTagger(object):

    def tag(self, parts):
        cwd = os.environ['SNORKELHOME'] + '/tutorials/babble/protein/geniatagger-3.0.2/'
        p = Popen(['./geniatagger -nt'], shell=True, cwd=cwd, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        words = u' '.join(parts['words']).encode('utf-8')
        out = p.communicate(input=words)[0]
        geniaOutputLines = out.decode('utf-8').split(u'\n')[4:-1]
        geniaWords = []
        geniaTags = []
        for line in geniaOutputLines:
            word = line.split(u'\t')[0]
            geniaWords.append(word)
            tag = line.split(u'\t')[-1]
            geniaTags.append(tag)

        for i, word in enumerate(parts['words']):
            tag = parts['entity_types'][i]
            geniaTag = geniaTags[i] if len(geniaTags) > i else 'O'
            geniaWord = geniaWords[i] if len(geniaWords) > i else '-'

            def bAndItoProtein(strToProcess=''):
                return 'protein' if strToProcess.find('protein') > -1 else strToProcess
            def checkForWordsAtIndex(strToFind, parentArray, index):
                numberOfWords = len(strToFind.split(' '))
                return True if ' '.join(parentArray[index:index+numberOfWords]).lower() == strToFind else False
            def checkCurrentForIndex(strToFind):
                return checkForWordsAtIndex(strToFind, parts['words'],i)
            #TODO write faster lookup
            def doesCurrentWordStartKinaseInList(listOfKinases):
                return reduce(lambda a,b: a or b, list(map(checkCurrentForIndex,listOfKinases)))

            kinaseList = [ 'pink1',
                           'lrrk2',
                           'leucine-rich repeat kinase 2',
                           'leucine rich repeat kinase 2',
                           'pten-induced putative kinase 1' ]

            if len(word) > 4 and (tag is None or tag is 'O'):
                wl = word.lower()
                if (doesCurrentWordStartKinaseInList(kinaseList)):
                    parts['entity_types'][i] = 'kinase'
                    parts['entity_cids'][i] = 'kinase'
                    parts['ner_tags'][i] = 'kinase'
                else:
                    parts['entity_types'][i] = bAndItoProtein(geniaTag)
                    parts['entity_cids'][i] = bAndItoProtein(geniaTag)
                    parts['ner_tags'][i] = bAndItoProtein(geniaTag)
        return parts

class ProteinKinaseLookupTagger (object):

    # default dict is unique list of all mouse and human genes
    # these represent proteins in the given text
    def __init__(self, fname='data/combined-protein-names.pkl'):
        cwd = os.environ['SNORKELHOME'] + '/tutorials/babble/protein/'
        with open('{}{}'.format(cwd,fname), 'rb') as f:
            self.protein_set = load(f)
            self.kinase_set = set([ 'pink1',
                           'lrrk2',
                           'jnk1',
                           'leucine-rich repeat kinase 2',
                           'leucine rich repeat kinase 2',
                           'pten-induced putative kinase 1' ])

    def tag(self, parts):
        for i, word in enumerate(parts['words']):
            tag = parts['entity_types'][i]
            if len(word) > 2 and tag in (None,'O'):
                wl = word.lower()
                # TODO determine whether populating entity_cids with dummy data
                # is necessary
                if wl in self.kinase_set:
                    parts['entity_types'][i] = 'kinase'
                elif wl.encode('utf-8') in self.protein_set:
                    parts['entity_types'][i] = 'protein'
        return parts
