import numpy as np

from image_objects import Point, BBox
# Point takes in x and y (left/right and top/bottom)

# Extractor Helper Functions
def extract_edge(bbox, side):
    coord = getattr(bbox, side)

    if side in ['top', 'bottom']:
        return Point(None, coord)
    elif side in ['left', 'right']:
        return Point(coord, None)
    else:
        raise Exception('Side is invalid')
        
        
def extract_corner(bbox, side1, side2):
    coord1 = getattr(bbox, side1)
    coord2 = getattr(bbox, side2)

    if side1 in ['top', 'bottom']:
        if side2 in ['left', 'right']:
            return Point(coord2, coord1)
        else:
            raise Exception('Side 2 is invalid')
            
    elif side1 in ['left', 'right']:
        if side2 in ['top', 'bottom']:
            return Point(coord1, coord2)
        else:
            raise Exception('Side 2 is invalid')
    else:
        raise Exception('Side 1 is invalid')

    
def extract_center(bbox):
    coordx = (getattr(bbox, 'left') + getattr(bbox, 'right'))/2.0
    coordy = (getattr(bbox, 'top') + getattr(bbox, 'bottom'))/2.0
    return Point(coordx, coordy)


def geoms_to_points(geoms, side):
    for i, geom in enumerate(geoms):
        if isinstance(geom, BBox):
            if side == 'center':
                geoms[i] = extract_center(geom)
            else:
                geoms[i] = extract_edge(geom, side)
    return geoms


  
# Point Comparison Helper Functions
def is_below(geom1, geom2):
    point1, point2 = geoms_to_points([geom1, geom2], 'bottom')
    if None in (point1.y, point2.y):
        raise Exception('Invalid Comparison')
    else:
        return point1.y > point2.y


def is_above(geom1, geom2):
    point1, point2 = geoms_to_points([geom1, geom2], 'top')
    return not is_below(point1, point2)


def is_right(geom1, geom2):
    point1, point2 = geoms_to_points([geom1, geom2], 'right')
    if None in (point1.x, point2.x):
        raise Exception('Invalid Comparison')
    else:
        return point1.x > point2.x


def is_left(geom1, geom2):
    point1, point2 = geoms_to_points([geom1, geom2], 'left')
    return not is_right(point1, point2)


def is_near(geom1, geom2, thresh=30.0):
    point1, point2 = geoms_to_points([geom1, geom2], 'center')
    coord1 = (point1.x, point1.y)
    coord2 = (point2.x, point2.y)
    
    if (None in coord1) or (None in coord2):
        raise Exception('Invalid Distance Comparison')
    else:
        dist = np.linalg.norm(np.array([point1.x - point2.x, point1.y - point2.y]))
        return dist <= thresh


def is_far(geom1, geom2):
    return not is_near(geom1, geom2, thresh=100.0)


# BBox Comparison Helper Functions
def is_smaller(bbox1, bbox2, mult=1.0):
    return bbox1.area() < bbox2.area() / mult

def is_larger(bbox1, bbox2, mult=1.0):
    return bbox1.area() > bbox2.area() * mult

def is_wider(bbox1, bbox2, mult=1.0):
    return bbox1.width > bbox2.width * mult

def is_taller(bbox1, bbox2, mult=1.0):
    return bbox1.height > bbox2.height * mult

def is_skinnier(bbox1, bbox2, mult=1.0):
    return bbox1.width < bbox2.width / mult

def is_shorter(bbox1, bbox2, mult=1.0):
    return bbox1.height < bbox2.height / mult

def is_overlaps(bbox1, bbox2, thresh=0.25):
    top = max(bbox1.top,bbox2.top)
    bottom = min(bbox1.bottom,bbox2.bottom)
    left = max(bbox1.left, bbox2.left)
    right = min(bbox1.right, bbox2.right)
    
    h = abs(top - bottom)
    w = abs(left - right)
    overlap_area = h*w
    
    return overlap_area >= thresh*max(bbox1.area(), bbox2.area())

def is_surrounds(bbox1, bbox2):
    return is_within(bbox2, bbox1)

def is_within(bbox1, bbox2):
    if bbox1.area() > bbox2.area():
        return False
    return (
        is_below(bbox2, bbox1) and 
        is_above(bbox2, bbox1) and 
        is_right(bbox2, bbox1) and 
        is_left(bbox2, bbox1))

    
def is_aligned(bbox1, bbox2, thresh=10.):
    thresh = 10
    raise NotImplementedError


helpers = {
    'extract_edge': extract_edge, 
    'extract_corner': extract_corner, 
    'extract_center': extract_center, 
    'is_below': is_below, 
    'is_above': is_above, 
    'is_right': is_right, 
    'is_left': is_left, 
    'is_near': is_near, 
    'is_far': is_far, 
    'is_smaller': is_smaller,
    'is_larger': is_larger,
    'is_wider': is_wider,
    'is_taller': is_taller,
    'is_skinnier': is_skinnier,
    'is_shorter': is_shorter,
    'is_aligned': is_aligned, 
    'is_overlaps': is_overlaps,
    'is_surrounds': is_surrounds,
    'is_within': is_within, 
}