from src.structure import Structure, decode_innovations
import numpy as np
from src.operations import upper_tri_masking, pad_with_zeros
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
    
    n1 = innov1.shape[0]
    n2 = innov2.shape[0]
    
    child = Structure(parent1._nodes[0:constrain_n], parent1.elastic_modulus, parent1._Fos_target, parent1._node_mass_k, parent1._yield_strenght, parent1.density, parent1._corner, parent1._crossover_radius, parent1._round_digit)
    child.init(parent1._nodes[constrain_n:])
    child._trusses = np.copy(parent1._trusses)
    
    genome1 = upper_tri_masking(innov1)
    genome2 = upper_tri_masking(innov2)
    
    k = max(genome1.shape[0], genome2.shape[0])

    genome1.resize((k))
    genome2.resize((k))
    
    common_idx = np.nonzero(np.in1d(genome1, genome2))[0]
    common = np.zeros(k, dtype=bool)
    common[common_idx] = True
    
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
    mass_1 = np.sum(s1._trusses[6]*s1.density)
    mass_2 = np.sum(s2._trusses[6]*s2.density)
    
    #print(mass_2-mass_1)
    return abs(K_mass*(mass_2-mass_1) + K_nodes*(len(s2._nodes) - len(s2._nodes)))

def sharing(population, index: int, threshold: float) -> int:
    s = population[index]
    sharing = 0
    for i in range(0, len(population)):
        if index != i:
            distance = get_distance(s, population[i])
            if distance < threshold:
                sharing = sharing + (1 - distance/threshold)

    return sharing
    
def mutate(s: Structure, node_pos_k, area_k, connection_k, add_node_k, delete_node_k, corner, area) -> Structure:
    
    min_x = corner[0]
    min_y = corner[1]
    max_x = corner[2]
    max_y = corner[3]
    k = [node_pos_k, area_k, connection_k, add_node_k, delete_node_k]
    mutation_type = np.random.choice([1, 2, 3, 4, 5], 1, k)[0]
    
    if mutation_type == 1:
        if s._n_constrain != len(s._nodes):
            i = random.randrange(s._n_constrain, len(s._nodes))
            new_x = random.uniform(min_x, max_x)
            new_y = random.uniform(min_y, max_y)
              
            s._nodes[i,0] = new_x
            s._nodes[i,1] = new_y
        return s
        
    elif mutation_type == 2:
        return area_mutation(s, area)
    elif mutation_type == 3:
        return connection_mutation(s)
    elif mutation_type == 4:
        new_x = random.uniform(min_x, max_x)
        new_y = random.uniform(min_y, max_y)
        s.add_node(new_x, new_y, area)
        return s
    elif mutation_type == 5:
        if s._n_constrain < len(s._nodes):
            idx = random.choice(range(s._n_constrain, len(s._nodes)))
            s.remove_node(idx)
            return s
    
    return s

def area_mutation(s: Structure, area = [0.001, 1]) -> Structure:
    new_area = random.uniform(area[0], area[1])
    i,j = np.nonzero(s._trusses[0])
    if len(i) > 0:
        ix = random.randrange(0, len(i)-1)
        
        r = i[ix]
        c = j[ix]
        
        s._trusses[1, r, c] = new_area
        s._trusses[1, c, r] = new_area
    
    return s
    
def connection_mutation(s: Structure) -> Structure:
    new_state = random.choice([0,1])
    i,j = np.nonzero(s._trusses[0])
    if len(i) > 0:
        ix = random.randrange(0, len(i)-1)
        
        r = i[ix]
        c = j[ix]
        
        s._trusses[0, r, c] = new_state
        s._trusses[0, c, r] = new_state
    
    return s