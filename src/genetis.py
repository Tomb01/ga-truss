from src.structure import Structure
import numpy as np
from src.operations import make_sym
from typing import List
import random

def crossover(parent1: Structure, parent2: Structure,  fit1: float, fit2: float) -> Structure:
    """
    Take 2 parent individuals and make a child with both parents characteristics

    Args:
        parent1 (Structure): 
        parent2 (Structure): 
        fit1 (float): fitness of parent 1
        fit2 (float): fitness of parent 2

    Returns:
        Structure: child structure
    """
    
    inn1 = parent1.get_node_innovations()
    inn2 = parent2.get_node_innovations()
    kp1 = fit1/(fit1+fit2)
    constrain_n = parent1._n_constrain

    n1 = len(inn1)
    n2 = len(inn2)
    
    filter_p1 = np.arange(n1)
    filter_p2 = np.zeros(n2, dtype=np.int32)
    j = n1 
    for i in range(0, n2):
        idx = np.where(inn1==inn2[i])[0]
        if len(idx)>0:
            # common gene
            filter_p2[i] = idx[0]
        else:
            # add to tail
            filter_p2[i] = j
            j = j+1
    
    k = j
            
    _, idx, c = np.unique(filter_p2, return_inverse=True, return_counts=True)
    fp1 = np.ix_(filter_p1, filter_p1)
    fp2 = np.ix_(filter_p2, filter_p2)

    genome1 = np.zeros((k, k))
    genome2 = np.zeros((k, k))

    genome1[fp1] = parent1.get_connections()
    genome2[fp2] = parent2.get_connections()
    
    child_conn_filter = make_sym(np.random.choice([0,1], k*k, p=[1-kp1, kp1]).reshape((k,k))) == 1
    child_genome = np.copy(genome1)
    child_genome[child_conn_filter] = genome2[child_conn_filter]
    np.fill_diagonal(child_genome, 0)
    
    # set child
    c = Structure(parent1._nodes[0:constrain_n], parent1._parameters)
    c.init(k)
    child_nodes = c._nodes
    child_nodes[filter_p2] = parent2._nodes
    child_nodes[filter_p1] = parent1._nodes
    c._nodes = child_nodes
    c.set_connections(child_genome)
    
    # Set areas
    area1 = np.zeros((k, k))
    area2 = np.zeros((k, k))
    area1[fp1] = parent1.get_area()
    area2[fp2] = parent2.get_area()
    child_area = np.copy(area1)
    child_area[child_conn_filter] = area2[child_conn_filter]
    np.fill_diagonal(child_area, 0)
    c.set_area(child_area)
    
    c.aggregate_nodes()
    c.healing()

    return c
    
def mutate(s: Structure, mutation_k: List[int], area_range) -> Structure:
    """
    Choose the mutation to apply based on mutation coefficients

    Args:
        s (Structure): original structure
        mutation_k (List[int]): list of mutation coefficients (See EvolutionParameter)
        area (_type_): area range for new bar (See StructureParameter)

    Returns:
        Structure: mutated structure
    """
    
    mutation_type = np.random.choice([1, 2, 3, 4, 5], p=mutation_k)
    
    if mutation_type == 1:
        return node_position_mutation(s)
    elif mutation_type == 2:
        return area_mutation(s, area_range)
    elif mutation_type == 3:
        return connection_mutation(s, area_range)
    elif mutation_type == 4:
        return add_node_mutation(s, area_range)
    elif mutation_type == 5:
        return remove_node_mutation(s)
    
    return s

def add_node_mutation(s: Structure, area_range) -> Structure:
    """
    Add random node to structure and link with at least 2 other nodes

    Args:
        s (Structure): original structure
        area (_type_): area range for new bars

    Returns:
        Structure: mutated structure
    """
    corner = s._parameters.corner
    new_x = random.uniform(corner.min_x, corner.max_x)
    new_y = random.uniform(corner.min_y, corner.max_y)
    s.add_node(new_x, new_y, area_range)
    return s

def remove_node_mutation(s: Structure) -> Structure:
    """
    Remove random node from a structure.
    Constrained node (from problem) will not be removed

    Args:
        s (Structure): original structure

    Returns:
        Structure: mutated structure
    """
    if s._n_constrain < len(s._nodes):
        idx = random.choice(range(s._n_constrain, len(s._nodes)))
        s.remove_node(np.array([idx]))
    return s

def area_mutation(s: Structure, area_range) -> Structure:
    """
    Change a random bar section area within a range

    Args:
        s (Structure): original structure
        area (_type_): `[min, max]` area range

    Returns:
        Structure: mutated structure
    """
    new_area = round(random.uniform(area_range[0], area_range[1]), s._parameters.round_digit)
    conn = s.get_connections()
    areas = s.get_area()
    i,j = np.nonzero(conn)
    if len(i) > 0:
        ix = random.randrange(0, len(i)-1)
        r = i[ix]
        c = j[ix]
        areas[r, c] = new_area
        areas[c, r] = new_area
        s.set_area(areas)
    
    return s

def node_position_mutation(s: Structure) -> Structure:
    """
    Change position to random node

    Args:
        s (Structure): original structure

    Returns:
        Structure: mutated structure
    """
    if s._n_constrain != len(s._nodes):
        corner = s._parameters.corner
        i = random.randrange(s._n_constrain, len(s._nodes))
        x = s._nodes[i,0]
        y = s._nodes[i,0]
        r = s._parameters.max_length/2
        new_x = random.uniform(x - r,  x + r)
        new_y = random.uniform(y - r,  y + r)
        
        if new_x > corner.max_x:
            new_x = corner.max_x
        elif new_x < corner.min_x:
            new_x = corner.min_x
            
        if new_y > corner.max_y:
            new_y = corner.max_y
        elif new_y < corner.min_y:
            new_y = corner.min_y
              
        s._nodes[i,0] = round(new_x, s._parameters.round_digit)
        s._nodes[i,1] = round(new_y, s._parameters.round_digit)
    return s
    
def connection_mutation(s: Structure, area_range) -> Structure:
    """
    Invert the status of nodes connection
    Example: if nodes was liked with a bar that bar is removed, otherwise a new bar will be created 

    Args:
        s (Structure): original structure
        area (_type_): `[min, max]` area range for new bar 

    Returns:
        Structure: mutated structure
    """
    random_idx = np.arange(len(s._nodes))
    np.random.shuffle(random_idx)
    r = random_idx[0]
    c = random_idx[1]
    
    conn = s.get_connections()
    areas = s.get_area()
    new_state = int(not conn[r, c])
    conn[r, c] = new_state
    conn[c, r] = new_state
    new_area = round(random.uniform(area_range[0], area_range[1]) * new_state, s._parameters.round_digit)
    areas[c, r] = new_area
    areas[r, c] = new_area

    s.set_connections(conn)
    s.set_area(areas)

    return s