from src.structure import Structure
import numpy as np
from src.operations import upper_tri_masking
import random

def crossover(parent1: Structure, parent2: Structure, constrain_n: int, fit1: float, fit2: float) -> Structure:
    # parent 1 is dominant
    parent1.set_innovations()
    parent2.set_innovations()
    if fit1 < fit2:
        # swamp parent
        swamp = parent1
        parent1 = parent2
        parent2 = swamp
        
    innov1 = parent1.get_innovations()
    innov2 = parent2.get_innovations()
    
    child = Structure(parent1._nodes[0:constrain_n], parent1._parameters)
    child.init(parent1._nodes[constrain_n:])
    child._trusses = np.copy(parent1._trusses)
    
    genome1 = upper_tri_masking(innov1)
    genome2 = upper_tri_masking(innov2)
    
    k = max(genome1.shape[0], genome2.shape[0])

    genome1.resize((k))
    genome2.resize((k))

    #print(genome1, genome2)
    
    common_idx = np.nonzero(np.in1d(genome1, genome2))[0]
    common = np.zeros(k, dtype=bool)
    common[common_idx] = True
    
    #print(common_idx)
    
    for i in range(0, k):
        if common[i]:
            # Common connection, get state and area random parent trusses adj
            p1_r, p1_c = np.argwhere(innov1 == genome1[i])[0]
            p2_r, p2_c = np.argwhere(innov2 == genome1[i])[0]
            state = random.choice([parent1._trusses[0, p1_r, p1_c], parent2._trusses[0, p2_r, p2_c]])
            area = random.choice([parent1._trusses[1, p1_r, p1_c], parent2._trusses[1, p2_r, p2_c]])
            
            #print(p1_r, p1_c, state, [parent1._trusses[0, p1_r, p1_c], parent2._trusses[0, p2_r, p2_c]])
            
            #print(state, p1_r, p1_c, [parent1._trusses[0, p1_r, p1_c], parent2._trusses[0, p2_r, p2_c]])
            child._trusses[0, p1_r, p1_c] = state
            child._trusses[0, p1_c, p1_r] = state
            child._trusses[1, p1_r, p1_c] = area
            child._trusses[1, p1_c, p1_r] = area
    
    return child
    
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
    
    mutation_type = np.random.choice([1, 2, 3, 4, 5], 1, mutation_k)[0]
    
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

    return s