from src.classes.truss import *
from src.classes.contrains import *
from src.classes.structure import *
import numpy as np
from math import isclose

if __name__ == "__main__":
    # Tema 15/06/2023 - A, es.1 
    node1 = TrussNode(2.0, 0.0)
    node2 = TrussNode(1, 0.0)
    node3 = TrussNode(0, 0)
    node4 = TrussNode(0, 1)
    node5 = TrussNode(0, -1)

    HingeConstrain(node1)
    HingeConstrain(node3)
    HingeConstrain(node4)
    HingeConstrain(node5)
            
    trusses = [Truss(node1, node2, 1.0, 1.0), Truss(node2, node4, 1.0, 1.0), Truss(node2, node3, 1.0, 1.0), Truss(node2, node5, 1.0, 1.0)]
    nodes = [node1, node2, node3, node4, node5]
    
    structure = Structure(nodes, trusses)
    
    A, b = structure.populate()
    #print(A[5])
    zeros = np.zeros(A.shape[1])
    j = 0
    b[1] = 1
    print(A.shape)
    print(np.linalg.matrix_rank(A), A.shape[0])
    np.set_printoptions(precision=3, suppress=True)
    print(np.linalg.solve(A, b).transpose())
    for row in A:
        t = input()
        print(row)
