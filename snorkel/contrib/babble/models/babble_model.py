from snorkel.annotations import LabelAnnotator
from snorkel_model import SnorkelModel, TRAIN, DEV, TEST

from snorkel.contrib.babble import Babbler

class BabbleModel(SnorkelModel):
    def babble(self, babbler, **kwargs):
        self.babbler = babbler
        self.babbler.apply(split=self.config['babbler_split'], 
                           parallelism=self.config['parallelism'])
        self.lfs = self.babbler.lfs

    def label(self):
        if not self.labeler:
            self.labeler = LabelAnnotator(lfs=self.lfs)  
        for split in self.config['splits']:
            if split == TEST:
                # No need for labels on test set
                continue
            else:
                # For now, just duplicate the labeling effort; re-lable all non-test splits
                # Label a new split
                num_candidates = self.session.query(self.candidate_class).filter(self.candidate_class.split == split).count()
                if num_candidates > 0:
                    L = SnorkelModel.label(self, self.labeler, split)
                    num_candidates, num_labels = L.shape
                    print("\nLabeled split {}: ({},{}) sparse (nnz = {})".format(split, num_candidates, num_labels, L.nnz))