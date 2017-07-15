import sys
import os
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'pymetamap'))
from pymetamap import MetaMap


class MetaMapAPI(object):

    def __init__(self, mm=MetaMap):
        self.mm = mm.get_instance(
            '/Users/morgism/Developer/Python/metamap/public_mm/bin/metamap16')

    def tag(self, sentence):
        sent = [sentence.get("text")]
        concepts, error = self.mm.extract_concepts(sent, [1])

        for concept in concepts:
            if concept.semtypes == "'[sosy]'":
                print "symptom!"
            elif concept.semtypes == "'[dsyn]'":
                print "disease!"

        return sentence
