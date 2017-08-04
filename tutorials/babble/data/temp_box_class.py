import numpy as np
import os
import sys

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
        