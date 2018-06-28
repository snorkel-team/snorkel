import numpy as np
import sys
import glob 

from util import lf_area, lf_eccentricity, lf_perimeter, lf_intensity, lf_ratio

# # add coral to path 
sys.path.append('/Users/vincentchen/code/coral') #adding Coral Home, temp hack
# sys.path.append('../') #adding Coral Home, temp hack
sys.path.append('/Users/vincentchen/code/numbskull') # for numbskull
sys.path.append('/Users/vincentchen/code/coral-experimental/venv/lib/python2.7') # for numba
print sys.path

from coral.learning import CoralModel
from coral.learning import CoralDependencySelector
from numbskull.udf import *
from coral.static_analysis.dependency_learning import find_dependencies

class PrimitiveObject(object):
    def save_primitive_matrix(self,primitive_mtx):
        self.primitive_mtx = primitive_mtx
        self.discrete_primitive_mtx = primitive_mtx
        self.num_primitives = np.shape(self.primitive_mtx)[1]
    
    def save_primitive_names(self,names):
        self.primitive_names = names
        if len(self.primitive_names) != self.num_primitives:
            Exception('Incorrect number of Primitive Names')
            
def create_primitives(vocab_matrix):
    m = 5
    primitive_mtx = np.zeros((num_examples, m))
    for i in range(num_examples):
        primitive_mtx[i, 0] = vocab_matrix[0, :][i] # area
        primitive_mtx[i, 1] = vocab_matrix[1, :][i] # eccentricity
        primitive_mtx[i, 2] = vocab_matrix[6, :][i] # perimeter
        primitive_mtx[i, 3] = vocab_matrix[8, :][i] # intensity
    
    primitive_mtx[:, 4] = primitive_mtx[:, 0]/(primitive_mtx[:, 2]**2.) # ratio
    P = PrimitiveObject()
    P.save_primitive_matrix(primitive_mtx)
    return P

def coalesce_vocab_matrices(vocab_matrix_paths, failed_indexes_paths):
    # coalesce all vocab matrices 
    vocab_matrix_list = []
    for v_m, f_i in zip(vocab_matrix_paths, failed_indexes_paths):
        print 'reading vocab matrix', v_m, 'and failed idx', f_i
        vocab_matrix = np.load(v_m)
        
        # set failed indexes to `nan`
        failed_idx = np.load(f_i)
        vocab_matrix[:, failed_idx] = np.nan
        
        vocab_matrix_list.append(vocab_matrix)
        print 'shape:', vocab_matrix.shape
        
    all_vocab_matrices = np.concatenate(vocab_matrix_list, axis=1)
        
    print '---\nconcatenated shape:', all_vocab_matrices.shape
    return all_vocab_matrices

def main():
    '''
    Command Line Args: 
    [input_path] -- path to vocab_matrix(es) and failed_indexes
    [output_dir] -- output directory for marginals 
    '''
    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    vocab_matrix_paths, failed_indexes_paths = [], []
    dir_keyword = 'dir:'
    if input_path.startswith(dir_keyword):
        input_path = input_path[len(dir_keyword):]
        vocab_matrix_paths = glob.glob(input_path + '*vocab_matrix.npy')
        failed_indexes_paths = glob.glob(input_path + '*failed_indexes.npy')
    else: 
        vocab_matrix_paths = [input_path + 'vocab_matrix.npy']
        failed_indexes_paths = [input_path + 'failed_indexes.npy']
    
    print 'read vocab_matric_paths:', vocab_matrix_paths
    print 'failed_indexes paths:', failed_indexes_paths

    full_vocab_matrix = coalesce_vocab_matrices(vocab_matrix_paths, failed_indexes_paths)

    P = create_primitives(full_vocab_matrix)
    primitive_names = ['area', 'eccentricity', 'perimeter', 'intensity', 'ratio']

    L_names = [lf_area, lf_eccentricity, lf_perimeter, lf_intensity, lf_ratio]
    L_deps = find_dependencies(L_names, primitive_names)
    print L_deps
    
    L = np.zeros((len(L_names), num_examples))
    for i in xrange(num_examples):
        for j in xrange(5):
            vocab_elems = P.primitive_mtx[i,L_deps[j]]
            L[j,i] = L_names[j](*vocab_elems)

if __name__ == '__main__':
    main()