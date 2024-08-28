from src.structure import Structure
import numpy as np
from src.operations import upper_tri_masking, make_sym
import random

def crossover(parent1: Structure, parent2: Structure, constrain_n: int, fit1: float, fit2: float) -> Structure:
    
    inn1 = parent1.get_node_innovations()
    inn2 = parent2.get_node_innovations()
    kp1 = fit1/(fit1+fit2)
    print(fit1, fit2)
    
    common = np.isin(inn1, inn2)
    common_idx = np.nonzero(common)[0]
    n1 = len(inn1)
    n2 = len(inn2)
    k = n1 + n2 - len(common_idx)

    genome = np.empty(k, dtype=np.object_)
    genome[0:n1] = inn1
    filter_p1 = np.arange(n1)
    filter_p2 = np.zeros(n2, dtype=np.int32)
    j = n1
    for i in range(0, n2):
        gene = inn2[i]
        idx = np.where(genome==gene)[0]
        if len(idx)>0:
            # common gene
            filter_p2[i] = idx[0]
        else:
            # add to tail
            filter_p2[i] = j
            j = j+1
            
    _, idx, c = np.unique(filter_p2, return_inverse=True, return_counts=True)
    if np.any(c>1):
        raise ValueError("Multiple join")
    
    fp1 = np.ix_(filter_p1, filter_p1)
    fp2 = np.ix_(filter_p2, filter_p2)

    genome1 = np.zeros((k, k))
    genome2 = np.zeros((k, k))

    genome1[fp1] = parent1._trusses[0]
    genome2[fp2] = parent2._trusses[0]
    
    ## TODO
    # Add node random delete
    # Add joint for substructure
    
    child_conn_filter = make_sym(np.random.choice([0,1], k*k, p=[kp1, 1-kp1]).reshape((k,k))) == 1
    child_genome = np.copy(genome1)
    child_genome[child_conn_filter] = genome2[child_conn_filter]
    
    # set child
    c = Structure(parent1._nodes[0:constrain_n], parent1._parameters)
    c.init(k-constrain_n)
    child_nodes = c._nodes
    child_nodes[filter_p2] = parent2._nodes
    child_nodes[filter_p1] = parent1._nodes
    c._nodes = child_nodes
    c._trusses[0] = child_genome
    c._trusses[1] = child_genome
    
    c.healing()
    
    return c
    
def get_distance(s1: Structure, s2: Structure, K_mass = 1, K_cm = 1, K_nodes = 1) -> float:
    mass_1 = np.sum(s1._trusses[6]*s1._parameters.material.density)
    mass_2 = np.sum(s2._trusses[6]*s2._parameters.material.density)
    
    #print(mass_2-mass_1)
    return abs(K_mass*(mass_2-mass_1) + K_nodes*(len(s2._nodes) - len(s2._nodes)))

def sharing(population, index: int, threshold: float) -> int:
    s = population[index]
    sharing = 0
    for i in range(0, len(population)):
        if index != i:
            distance = abs(s - population[i]) #get_distance(s, population[i])
            if distance < threshold:
                sharing = sharing + 1

    return sharing
    
def mutate(s: Structure, mutation_k, area) -> Structure:
    
    mutation_type = np.random.choice([1, 2, 3, 4, 5], p=mutation_k)
    
    if mutation_type == 1:
        return node_position_mutation(s)
    elif mutation_type == 2:
        return area_mutation(s, area)
    elif mutation_type == 3:
        return connection_mutation(s, area)
    elif mutation_type == 4:
        return add_node_mutation(s, area)
    elif mutation_type == 5:
        return remove_node_mutation(s)
    
    return s

def add_node_mutation(s: Structure, area) -> Structure:
    corner = s._parameters.corner
    new_x = random.uniform(corner.min_x, corner.max_x)
    new_y = random.uniform(corner.min_y, corner.max_y)
    s.add_node(new_x, new_y, area)
    return s

def remove_node_mutation(s: Structure) -> Structure:
    if s._n_constrain < len(s._nodes):
        idx = random.choice(range(s._n_constrain, len(s._nodes)))
        s.remove_node(idx)
    return s

def area_mutation(s: Structure, area) -> Structure:
    new_area = round(random.uniform(area[0], area[1]), s._parameters.round_digit)
    i,j = np.nonzero(s._trusses[0])
    if len(i) > 0:
        ix = random.randrange(0, len(i)-1)
        
        r = i[ix]
        c = j[ix]
        
        s._trusses[1, r, c] = new_area
        s._trusses[1, c, r] = new_area
    
    return s

def node_position_mutation(s: Structure) -> Structure:
    if s._n_constrain != len(s._nodes):
        corner = s._parameters.corner
        i = random.randrange(s._n_constrain, len(s._nodes))
        new_x = random.uniform(corner.min_x, corner.max_x)
        new_y = random.uniform(corner.min_y, corner.max_y)
              
        s._nodes[i,0] = round(new_x, s._parameters.round_digit)
        s._nodes[i,1] = round(new_y, s._parameters.round_digit)
    return s
    
def connection_mutation(s: Structure, area) -> Structure:
    random_idx = np.arange(len(s._nodes))
    np.random.shuffle(random_idx)
    #print(random_idx)
    r = random_idx[0]
    c = random_idx[1]
        
    new_state = int(not s._trusses[0, r, c])
    s._trusses[0, r, c] = new_state
    s._trusses[0, c, r] = new_state
    new_area = round(random.uniform(area[0], area[1]) * new_state, s._parameters.round_digit)
    s._trusses[0, c, r] = new_area
    s._trusses[0, r, c] = new_area


    return s