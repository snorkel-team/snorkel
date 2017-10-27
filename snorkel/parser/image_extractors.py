import glob
import os

import numpy as np

from ..models import Candidate, Context, Image, Bbox
from ..udf import UDF, UDFRunner

"""
TODO: 
This implementation is wasteful in the sense that we need to flush between every
stage (Image extraction, Bbox extraction, Candidate extraction). Instead, 
separate this into the normel Snorkel pipeline pieces: a document (image) 
preprocessor, a corpus (bboxes) parser, and a candidate extractor.

Things to think about:
We currently don't care about images that don't have pairs of person/bikes,
    so extracting all images first may end up adding a bunch of Image objects
    we will never use.
CandidateSpace could be 1 bounding box at a time
Matchers could be super simple check of category_id
Throttler would be needed to require that the two Bboxes overlap
"""

class ImageCorpusExtractor(UDFRunner):

    def __init__(self, candidate_class):
        super(ImageCorpusExtractor, self).__init__(ImageCorpusExtractorUDF,
                                                   candidate_class=candidate_class)

    def clear(self, session, **kwargs):
        session.query(Context).delete()
        # We cannot cascade up from child contexts to parent Candidates,
        # so we delete all Candidates too
        session.query(Candidate).delete()


class ImageCorpusExtractorUDF(UDF):

    def __init__(self, candidate_class, **kwargs):
        super(ImageCorpusExtractorUDF, self).__init__(**kwargs)
        self.candidate_class = candidate_class

    def apply(self, x, person_id=[1], object_id=[2], **kwargs):
        # TEMP: overly specific here to bike task
        ann, image_idx, source = x
        
        person_indices = [i for i, box in enumerate(ann) if box['category_id'] in person_id]
        bike_indices = [i for i, box in enumerate(ann) if box['category_id'] in object_id]
        person_bike_tuples = [(x,y) for x in person_indices for y in bike_indices]
        valid_pairs = get_valid_pairs(ann, person_bike_tuples)

        if valid_pairs:
            # Make Image objects
            name = "{}:{}".format(source, image_idx)
            stable_id = "{}::image".format(name)
            image = Image(name=name, stable_id=stable_id)
            self.session.add(image)
            self.session.flush()

            # Make Bbox objects
            bboxes = [None] * len(ann)
            for bbox_idx in set(person_indices + bike_indices):
                x, y, w, h = ann[bbox_idx]['bbox']
                category_id = ann[bbox_idx]['category_id']
                stable_id = "{}::bbox:{}".format(name, bbox_idx)
                bbox = Bbox(stable_id=stable_id,
                            image=image,
                            position=bbox_idx,
                            category=category_id,
                            top=y,
                            bottom=y + h,
                            left=x,
                            right=x + w)
                self.session.add(bbox)
                bboxes[bbox_idx] = bbox                            
            self.session.flush()

            # Make Candidate objects
            for i, j in valid_pairs:
                args = (bboxes[i], bboxes[j])
                candidate_args = {'split': source} # For now, use source as split [train = 0, val = 1]
                for i, arg_name in enumerate(self.candidate_class.__argnames__):
                    candidate_args[arg_name + '_id'] = args[i].id
                yield self.candidate_class(**candidate_args)


def get_valid_pairs(anns, tuples):
    valid_pairs = []
    for person, bike in tuples:
        person_box = anns[person]['bbox']
        bike_box = anns[bike]['bbox']
        
        #Temp hack to add all images in anns files
        valid_pairs.append((person,bike))

        #if overlap(person_box, bike_box):
        #    valid_pairs.append((person,bike))

    
    return valid_pairs

def overlap(box1, box2):
    if (box1[0] + box1[2] < box2[0] or 
        box2[0] + box2[2] < box1[0] or 
        box1[1] + box1[3] < box2[1] or 
        box2[1] + box2[3] < box1[1]):
        return False
    else:
        return True


class ImagePreprocessor(object):
    """
    Processes a file or directory of files into a set of Document objects.

    :param path: filesystem path to file or directory to parse
    :param max_docs: the maximum number of Documents to produce,
        default=float('inf')

    """

    def __init__(self, path, source=0, max_docs=float('inf')):
        self.path = path
        self.source = source
        self.max_docs = max_docs

    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.

        """
        image_count = 0
        for fp in self._get_files(self.path):
            file_name = os.path.basename(fp)
            if self._can_read(file_name):
                for img in self.parse_file(fp, file_name):
                    yield img
                    image_count += 1
                    if self.max_docs and image_count >= self.max_docs:
                        return

    def __iter__(self):
        return self.generate()

    def get_stable_id(self, doc_id):
        return "%s::image:0:0" % doc_id

    def parse_file(self, fp, file_name):
        raise NotImplementedError()

    def _can_read(self, fpath):
        return True

    def _get_files(self, path):
        if os.path.isfile(path):
            fpaths = [path]
        elif os.path.isdir(path):
            fpaths = [os.path.join(path, f) for f in os.listdir(path)]
        else:
            fpaths = glob.glob(path)
        if len(fpaths) > 0:
            return fpaths
        else:
            raise IOError("File or directory not found: %s" % (path,))


class CocoPreprocessor(ImagePreprocessor):
    def parse_file(self, fp, file_name):
        anns = np.load(fp).tolist()
        for i, ann in enumerate(anns):
            yield ann, i, self.source

    def _can_read(self, fpath):
        return fpath.endswith('.npy')