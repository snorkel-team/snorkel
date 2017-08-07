from snorkel.contrib.babble import Babbler

from snorkel_model import SnorkelModel

class BabbleModel(SnorkelModel):
    def babble(self, babbler, **kwargs):
        self.babbler = babbler
        self.babbler.apply(split=self.config['babbler_split'], 
                           parallelism=self.config['parallelism'])
        # apply filters here

