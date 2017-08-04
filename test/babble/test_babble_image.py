import os
import unittest

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.contrib.babble import SemanticParser
import image_explanations
# from pycocotools.coco import COCO

from temp_image_class import *

class TestBabbleImages(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # vg_folder = '/dfs/scratch0/paroma/visual_genome/'
        # coco_path = '/dfs/scratch0/paroma/coco/annotations/'
        
        # #Get annotations from .json file
        # def generate_anns(filename, coco_ids):
        #     coco=COCO(filename)
        #     catIds = coco.getCatIds(catNms=['person','bicycle']);
        #     set_anns = []

        #     for set_id in coco_ids:
        #         annIds = coco.getAnnIds(imgIds=set_id, catIds=catIds, iscrowd=None)
        #         anns = coco.loadAnns(annIds)

        #         temp_list = []
        #         for i in xrange(len(anns)):
        #             temp_dict = {'category_id':anns[i]['category_id'], 'bbox':anns[i]['bbox']}
        #             temp_list.append(temp_dict)
        #         set_anns.append(temp_list)

        #     return set_anns
        
        # self.train_anns = generate_anns(coco_path+'instances_train2014.json', train_mscoco)
        cls.train_mscoco = np.load('./train_mscoco.npy')
        train_anns = np.load('./train_anns.npy')
        cls.train_anns = train_anns.tolist()
        
        cls.sp = SemanticParser(mode='image', beam_width=10, top_k=-1)

    @classmethod
    def tearDownClass(cls):
        pass
    
    def check_explanations(self, explanations):
        self.assertTrue(len(explanations))
        for e in explanations:
            if e.candidate and not isinstance(e.candidate, tuple):
                e.candidate = Image_Candidate(idx=e.candidate,coco_ids=self.train_mscoco,coco_anns=self.train_anns)
            LF_dict = self.sp.parse_and_evaluate(e, show_erroring=True) # show_nothing=True
            # parses = self.sp.parse(e, return_parses=True)
            if e.semantics:
                self.assertTrue(len(LF_dict['correct']) > 0)
            else:
                self.assertTrue(len(LF_dict['passing']) > 0)
            self.assertTrue(len(LF_dict['correct']) + len(LF_dict['passing']) <= 3)

    def test_boxes(self):
        self.check_explanations(image_explanations.boxes)


if __name__ == '__main__':
    unittest.main()