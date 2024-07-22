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
    
    # Get common nodes
    common_nodes_idx = np.nonzero(np.in1d(nodes1, nodes2))[0]
    common_nodes = np.zeros(n, dtype=bool)
    common_nodes[common_nodes_idx] = True
    print(common_nodes, common_nodes_idx)
    
    genome1 = upper_tri_masking(innov1)
    genome2 = upper_tri_masking(innov2)
    
    k = max(genome1.shape[0], genome2.shape[0])

    genome1.resize((k))
    genome2.resize((k))
    
    common_idx = np.nonzero(np.in1d(genome1, genome2))[0]
    common = np.zeros(k, dtype=bool)
    common[common_idx] = True
    
    dominant = np.copy(genome1)
    nodes = parent1._nodes
    trusses = parent1._trusses
    
    for i in range(0, k):
        if common[i]:
            # Common connection, get state and area random parent trusses adj
            inn_row = i // (n1-1)
            inn_col = i % (n1-1) + 1
            state = random.choice([parent1._trusses[0,inn_row,inn_col], parent2._trusses[0,inn_row,inn_col]])
            if state == 1:
                area = random.choice([parent1._trusses[0,inn_row,inn_col], parent2._trusses[0,inn_row,inn_col]])
                trusses[0,inn]
            print(state, area)
    
    child = Structure(nodes[0:constrain_n], parent1.elastic_modulus)
    child._nodes = nodes
    child._trusses = trusses
    
    return child
            