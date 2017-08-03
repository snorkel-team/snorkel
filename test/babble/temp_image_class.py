import numpy as np
import os
import sys

sys.path.append('/dfs/scratch0/paroma/coco')
sys.path.append('/dfs/scratch0/paroma/coco/PythonAPI/')

from pycocotools.coco import COCO
import skimage.io as io
import pylab
import cv2


class Image_Candidate(object):
    
    def __init__(self, idx=-1,coco_ids=train_mscoco,coco_anns=train_anns):
        if idx <= -1 or idx >= 903 :
            print 'Invalid Train Image Index'
        
        self.idx = idx
        self.coco_ann = coco_anns[idx] #CURRENT IMAGE ANNOTATIONS
        self.original_url = 'http://mscoco.org/images/'+ str(coco_ids[idx])
        self.annotated_url = 'http://paroma.github.io/turk_images/train_' + str(idx)
        
        def generate_boxes(self):
            bboxes = []
            for i in xrange(len(self.coco_ann)):
                bbox_ann = self.coco_ann[i]
                bbox_curr = BBox(bbox_ann,coco_ids[idx])
                bboxes.append(bbox_curr)
            return bboxes
        self.bboxes = generate_boxes(self)
        
class BBox(object):
    
    def __init__(self, attr_dict, idx):
        self.parent_idx = idx
        
        x,y,w,h = attr_dict['bbox']
        label = attr_dict['category_id']
        
        self.width = w
        self.height = h
        
        self.top = y
        self.left = x
        self.bottom = self.top+h
        self.right = self.left+w
        
        def area(self):
            return self.width*self.height
        
        def perimeter(self):
            return 2*(self.width+self.height)
        
        def top_edge(self):
            return self.top
            
        def bottom_edge(self):
            pass
        
        def left_edge(self):
            pass
        
        def right_edge(self):
            pass
        
        def bottom_edge(self):
            pass
        