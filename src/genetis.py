from src.structure import Structure, decode_innovations
import numpy as np
from src.operations import upper_tri_masking, pad_with_zeros
import random

def crossover(parent1: Structure, parent2: Structure, constrain_n: int, fit1: float, fit2: float) -> Structure:
    # parent 1 is dominant
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
    child._trusses = parent1._trusses
    
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
            
            #print(state, p1_r, p1_c, [parent1._trusses[0, p1_r, p1_c], parent2._trusses[0, p2_r, p2_c]])
            child._trusses[0, p1_r, p1_c] = state
            child._trusses[0, p1_c, p1_r] = state
            child._trusses[1, p1_r, p1_c] = area
            child._trusses[1, p1_c, p1_r] = area
        else:
            if fit1 == fit2:
                print("")
    
    return child

def fit(k_tuple) -> float:
    k_structure, k_mass, k_Fos = k_tuple
    if k_structure == 0:
        return 0
    else:
        return k_structure * 1/(k_mass + k_Fos)