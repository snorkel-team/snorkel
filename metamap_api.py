import sys
import os
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'pymetamap'))
from pymetamap import MetaMap

DISEASE = "Disease"
SYMPTOM = "Symptom"
MetaMap_SYMPTOM = '[sosy]'
MetaMap_DISEASE = '[dsyn]'


class MetaMapAPI(object):

    def __init__(self, mm=MetaMap):
        self.mm = mm.get_instance(
            '/Users/morgism/Developer/Python/metamap/public_mm/bin/metamap16')

    def tag(self, sentence):
        sentence_text = [sentence.get("text")]
        sentence_text[0] = sentence_text[0].encode('ascii', errors='ignore')
        concepts, error = self.mm.extract_concepts(sentence_text, [1])

        for concept in concepts:
            if concept.semtypes == MetaMap_SYMPTOM:
                sentence = self.generate_entities(sentence, concept, SYMPTOM)
            elif concept.semtypes == MetaMap_DISEASE:
                sentence = self.generate_entities(sentence, concept, DISEASE)

        return sentence

    def generate_entities(self, sentence, concept, tag):
        position_information = concept.pos_info.split('/')
        # MetaMap counts the quotation starting at index 0
        # while CoreNLP does not, therefore we are left with an
        # off by one error.
        disease_character_start = int(position_information[0]) - 1
        disease_length = int(position_information[1])

        for index, character_offset in enumerate(sentence['char_offsets']):
            print('index: {0}'.format(index))
            print('character_offset: {0}'.format(character_offset))
