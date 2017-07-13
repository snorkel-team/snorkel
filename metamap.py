from pymetamap.pymetamap import MetaMap


class MetaMapAPI(object):

    def __init__(self):
        self.mm = MetaMap.get_instance(
            '/Users/morgism/Developer/Python/metamap/public_mm/bin/metamap16')

    def tag(self, parts):
        sents = ['John had a heart attack']
        concepts, error = self.mm.extract_concepts(sents, [1])
        print"Concepts: {0}".format(concepts)
        print"Error: {0}".format(error)

        return parts
