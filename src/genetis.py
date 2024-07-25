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
    n = max(n1, n2)
    
    child = Structure(parent1._nodes[0:constrain_n], parent1.elastic_modulus)
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
        else:
            if fit1 == fit2:
                node_mutation(child)
    
    return child
    
def get_compatibility(structure: Structure, C1=1, C3=1) -> float:
    w_mean = np.mean(structure._trusses[1])
    n = len(structure._nodes)
    e_count = n - structure._n_constrain
    return C1*e_count/n + C3*w_mean

def is_shared(c1: float, c2: float, t: float) -> int:
    if c1-t <= c2 <= c1+t:
        return 1
    else:
        return 0
    
def mutate(s, mutation_rate, k1, k2, k3, k4, max_rep = 2, area = [0.001, 1]) -> Structure:
    
    k = [k1, k2, k3, k4]
    mutation_type = np.random.choice([1, 2, 3, 4], 1, k)[0]
    mutate = np.random.choice([0,1], 1, [1-mutation_rate, mutation_rate])[0]
    
    #print(mutate, mutation_type)
    
    if mutate:
        if mutation_type == 1:
            return node_mutation(s, max_rep)
        elif mutation_type == 2:
            return area_mutation(s, area)
    
    return s
    
def node_mutation(s: Structure, max_rep = 2) -> Structure:
    if s._n_constrain != len(s._nodes):
        i = random.randrange(s._n_constrain, len(s._nodes))
        new_x = random.uniform(-max_rep, max_rep)
        new_y = random.uniform(-max_rep, max_rep)
        
        s._nodes[i,0] = new_x
        s._nodes[i,1] = new_y
    
    return s

def area_mutation(s: Structure, area = [0.001, 1]) -> Structure:
    new_area = random.uniform(area[0], area[1])
    i,j = np.nonzero(s._trusses[0])
    ix = random.randrange(0, len(i)-1)
    
    r = i[ix]
    c = j[ix]
    
    s._trusses[1, r, c] = new_area
    s._trusses[1, c, r] = new_area
    
    #print(s._trusses[1], r, c)
    
    return s