import numpy as np

class PrimitiveObject(object):

    def save_primitive_matrix(self,primitive_mtx):
        self.primitive_mtx = primitive_mtx
        self.discrete_primitive_mtx = primitive_mtx
        self.num_primitives = np.shape(self.primitive_mtx)[1]
    
    def save_primitive_names(self,names):
        self.primitive_names = names
        if len(self.primitive_names) != self.num_primitives:
            Exception('Incorrect number of Primitive Names')

    

def bike_human_nums(object_names):
    names = object_names.split(' , ')[1:]
    num_person = 0
    num_bicycles = 0
    for i in range(np.shape(names)[0]):
        name = names[i]
        if ('person' in name) or ('man' in name) or ('woman' in name) or ('girl' in name) or ('boy' in name) or ('people' in name):
            num_person = num_person+1
        if ('cycle' in name) or ('bike' in name) or ('bicycle' in name):
            num_bicycles = num_bicycles+1
    
    if (num_bicycles == 0) or (num_person == 0):
        return 0

    if num_person == num_bicycles:
        return 2
    elif num_person <= num_bicycles:
        return 0
    else:
        return 1


def bike_human_distance(object_names, object_x, object_y):
    names = object_names.split(' , ')[1:]
    person_position = np.array([[0,0],[0,0]])
    bicycle_position = np.array([[0,0],[0,0]])
    for i in range(np.shape(names)[0]):
        name = names[i]
        if ('person' in name) or ('man' in name) or ('woman' in name) or ('girl' in name) or ('boy' in name) or ('people' in name):
            person_position = np.concatenate((person_position, np.array([[object_x[i],object_y[i]]])))
        if ('cycle' in name) or ('bike' in name) or ('bicycle' in name):
            bicycle_position = np.concatenate((bicycle_position, np.array([[object_x[i],object_y[i]]])))
    
    person_position = person_position[2:,:]
    bicycle_position = bicycle_position[2:,:]
    
    if (np.shape(bicycle_position)[0] == 0) or (np.shape(person_position)[0] == 0):
        return -1
        
    import itertools
    if len(bicycle_position) >= len(person_position):
        list1 = [list(coord) for coord in bicycle_position]
        list2 = [list(coord) for coord in person_position]
    else:
        list2 = [list(coord) for coord in bicycle_position]
        list1 = [list(coord) for coord in person_position]
        
    coord_comb = [list1, list2]
    person_bike_pairs = itertools.product(*coord_comb)
    
    dists = []
    for pair in person_bike_pairs:
        for coord1, coord2 in pair:
            dists.append(np.linalg.norm(coord1-coord2))     
    
    return np.min(dists)





def bike_human_size(object_names, object_area):
    names = object_names.split(' , ')[1:]
    person_area = np.array([0])
    bicycle_area = np.array([0])
    
    
    for i in range(np.shape(names)[0]):
        name = names[i]
        if ('person' in name) or ('man' in name) or ('woman' in name) or ('girl' in name) or ('boy' in name) or ('people' in name):
            person_area = np.concatenate((person_area, [object_area[i]]))
        if ('cycle' in name) or ('bike' in name) or ('bicycle' in name):
            bicycle_area = np.concatenate((bicycle_area, [object_area[i]]))
    
    person_area = person_area[1:]
    bicycle_area = bicycle_area[1:]
    
    if (np.shape(bicycle_area)[0] == 0) or (np.shape(person_area)[0] == 0):
        area_diff = -1
    
    area_diff = -1
    for i in range(np.shape(bicycle_area)[0]):
        try:
            area_diff_temp = np.max((np.abs(bicycle_area[i]-person_area[:])))
            area_diff = np.max(area_diff_temp, area_diff)
        except:
            continue
    
    return area_diff