from image_objects import Point
#Point takes in x and y (left/right and top/bottom)

def extract_edge(bbox, side):
    coord = getattr(bbox, side)

    if side in ['top','bottom']:
        return Point(None,coord)
    elif side in ['left','right']:
        return Point(coord,None)
    else:
        raise Exception('Side is invalid')
        
        
def extract_corner(bbox, side1, side2):
    coord1 = getattr(bbox, side1)
    coord2 = getattr(bbox, side2)

    if side1 in ['top','bottom']:
        if side2 in ['left','right']:
            return Point(coord2,coord1)
        else:
            raise Exception('Side 2 is invalid')
            
    elif side1 in ['left','right']:
        if side2 in ['top','bottom']:
            return Point(coord1,coord2)
        else:
            raise Exception('Side 2 is invalid')
    else:
        raise Exception('Side 1 is invalid')

        
def extract_center(bbox):
    coordx = (getattr(bbox,'left') + getattr(bbox,'right')/2.
    coordy = (getattr(bbox,'top') + getattr(bbox,'bottom')/2.
    return Point(coordx,coordy)