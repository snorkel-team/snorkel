''' Helper functions for segmentation. 
(Anything that needs to be tuned goes into this file).'''

import numpy as np

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

def find_target_region(region_label, image):
    ''' 
    Given region labels, extracts region of interest using the following steps: 
        1) filter out regions that are too small 
        2) filter out regions with absurdly large eccentricity (image artifcats)
        3) narrow down to 2 brightest regions 
        4)  a) if regions are close to one another, pick one to right
            b) else, pick region closest to bottom left corner
            
    TUNEABLE PARAMS:
        - WIDTH_DELTA_THRESHOLD: num horizontal pixels between regions 
                to be considered 'close'
        - HEIGHT_DELTA_THRESHOLD: num vertical pixels between regions 
                to be considered 'close'
        - 
    '''
    
    # tune these params 
    WIDTH_DELTA_THRESHOLD = 25
    HEIGHT_DELTA_THRESHOLD = 15
    AREA_THRESHOLD = 30 
    ECCENTRICITY_THRESHOLD = 0.98
    
    def two_regions_close(two_regions):
        ''' Helper fn to determine if regions are considered 'close' '''
        
        if len(two_regions) != 2: return False
        
        width_delta = abs(two_regions[0].centroid[1] - two_regions[1].centroid[1])
        height_delta = abs(two_regions[0].centroid[0] - two_regions[1].centroid[0])
        return width_delta < WIDTH_DELTA_THRESHOLD \
                and height_delta < HEIGHT_DELTA_THRESHOLD
        
    regions = regionprops(region_label.astype(int), image)

    # remove all blobs < 30 area and > 0.95 eccentricity
    filtered = list(filter(lambda x: x.area >= AREA_THRESHOLD 
            and x.eccentricity < ECCENTRICITY_THRESHOLD, regions))
    
    if len(filtered) == 0: 
        raise ValueError('bad threshold', 
                [(r.area, r.eccentricity) for r in regions])
    
    # pick top 2 mean_intensity 
    sorted_mean_intensity = sorted(filtered, key=lambda x: -x.mean_intensity)[:2]
    
    if two_regions_close(sorted_mean_intensity):
        # return right most
        return sorted(sorted_mean_intensity, key=lambda x: -x.centroid[1])[0]
    else: 
        # return botom-left most
        bottom_left = np.array([image.shape[0], 0])
        bottom_leftmost = sorted(sorted_mean_intensity, key=lambda x: np.linalg.norm(bottom_left - x.centroid))[0]

        return bottom_leftmost
    
def preprocess_and_extract_region_label(image):
    try: 
        thresh = threshold_otsu(image)    
    except: 
        raise ValueError('invalid image for otsu thresholding', np.max(image))
    
    bw = closing(image > thresh, square(2))        
    region_label = label(bw, connectivity=2)
    return region_label 


def lf_area(area):
    if area >= 134:
        return -1 
    if area <= 65: 
        return 1
    return 0

def lf_eccentricity(eccentricity):
    if eccentricity >= 0.78: 
        return 1 
    if eccentricity <= 0.63:
        return -1  
    return 0
        
def lf_perimeter(perimeter):
    if perimeter <= 31: 
        return 1 
    return 0
    
def lf_intensity(intensity):
    if intensity >= 168: 
        return 1
    if intensity <= 108: 
        return -1
    return 0

def lf_ratio(ratio):
    if ratio >= 0.072: 
        return -1
    if ratio <= 0.062:
        return 1
    return 0