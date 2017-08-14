from snorkel.annotations import LabelAnnotator, load_gold_labels
from snorkel_model import SnorkelModel, TRAIN, DEV, TEST

from snorkel.contrib.babble import Babbler

class BabbleModel(SnorkelModel):
    def babble(self, babbler, **kwargs):
        self.babbler = babbler
        self.babbler.apply(split=self.config['babbler_split'], 
                           parallelism=self.config['parallelism'])
        self.lfs = self.babbler.lfs
        self.labeler = LabelAnnotator(lfs=self.lfs)

    def label(self, config=None, split=None):
        if config:
            self.config = config
        if not self.labeler:
            self.labeler = LabelAnnotator(lfs=self.lfs)  
        splits = [split] if split else self.config['splits']
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
                    if self.config['display_accuracies'] and split == DEV:
                        L_gold_dev = load_gold_labels(self.session, annotator_name='gold', split=1)
                        print(L.lf_stats(self.session, labels=L_gold_dev))
        