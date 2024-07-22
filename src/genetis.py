from src.structure import Structure
import numpy as np
from src.operations import upper_tri_masking, pad_with_zeros
import random

def crossover(parent1: Structure, parent2: Structure, constrain_n: int) -> Structure:
    # parent 1 is dominant
    innov1, nodes1 = parent1.get_innovations()
    innov2, nodes2 = parent2.get_innovations()
    
    n1 = innov1.shape[0]
    n2 = innov2.shape[0]
    n = max(n1, n2)
    
    nodes = parent1._nodes
    trusses = parent1._trusses
    
    genome1 = upper_tri_masking(innov1)
    genome2 = upper_tri_masking(innov2)
    
    k = max(genome1.shape[0], genome2.shape[0])

    genome1.resize((k))
    genome2.resize((k))
    
    common_idx = np.nonzero(np.in1d(genome1, genome2))[0]
    common = np.zeros(k, dtype=bool)
    common[common_idx] = True
    
    print(common)
    
    return 
            