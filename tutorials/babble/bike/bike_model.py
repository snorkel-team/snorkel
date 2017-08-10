import os
import numpy as np

from snorkel.parser import ImageCorpusExtractor, CocoPreprocessor
from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels

from snorkel.contrib.babble import Babbler
from snorkel.contrib.babble.models import BabbleModel

from tutorials.babble.bike import load_external_labels

class BikeModel(BabbleModel):

    def parse(self, anns_path):
        self.anns_path = anns_path
        train_path = anns_folder + 'train_anns.npy'
        val_path = anns_folder + 'val_anns.npy'

        corpus_extractor = ImageCorpusExtractor(candidate_class=Biker)

        coco_preprocessor = CocoPreprocessor(train_path, source=0)
        corpus_extractor.apply(coco_preprocessor)

        coco_preprocessor = CocoPreprocessor(val_path, source=1)
        corpus_extractor.apply(coco_preprocessor, clear=False)

        for split in [0, 1]:
            num_candidates = session.query(Biker).filter(Biker.split == split).count()
            print("Split {} candidates: {}".format(split, num_candidates))

    def get_candidates(self):
        

    def load_gold(self, anns_path=None, annotator_name='gold'):
        if anns_path:
            self.anns_path = anns_path
        labels_by_candidate = np.load(
            self.anns_path + 'labels_by_candidate.npy').tolist()

        for candidate_hash, label in labels_by_candidate.items():
            set_name, image_idx, bbox1_idx, bbox2_idx = candidate_hash.split(':')
            source = {'train': 0, 'val': 1}[set_name]
            stable_id_1 = "{}:{}::bbox:{}".format(source, image_idx, bbox1_idx)
            stable_id_2 = "{}:{}::bbox:{}".format(source, image_idx, bbox2_idx)
            context_stable_ids = "~~".join([stable_id_1, stable_id_2])
            query = self.session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
            query = query.filter(StableLabel.annotator_name == annotator_name)
            label = 1 if label else -1
            if query.count() == 0:
                self.session.add(StableLabel(
                    context_stable_ids=context_stable_ids,
                    annotator_name=annotator_name,
                    value=label))

        self.session.commit()
        reload_annotator_labels(self.session, self.candidate_class, 
            annotator_name, split=1, filter_label_split=False)


    def babble(self, explanations, user_lists={}, **kwargs):
        babbler = Babbler(mode='image', candidate_class=None, explanations=explanations)
        super(BikeModel, self).babble(babbler, **kwargs)