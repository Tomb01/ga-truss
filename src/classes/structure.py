from typing import List, Dict
import numpy as np
from src.classes.truss import Truss, TrussNode
from math import cos, sin

class Structure:

    _nodes: List[TrussNode]
    _trusses: List[Truss]
    _systemA: np.array

    def __init__(self, nodes: List[TrussNode], trusses: List[Truss]) -> None:
        self._nodes = nodes
        self._trusses = trusses

        self.check()
        self.assing_index()

    def check(self) -> bool:
        node_check: Dict[str, bool] = {}
        for truss in self._trusses:
            node = truss.get_nodes()
            node_check[node[0].get_id()] = True
            node_check[node[1].get_id()] = True

        node_list: Dict[str, bool] = {}
        for node in self._nodes:
            node_list[node.get_id()] = True

        #print(node_check, node_list)

        if node_list == node_check:
            return True
        else:
            return False
        
    def assing_index(self) -> None:
        for index in range(0, len(self._nodes)):
            self._nodes[index-0].set_index(index)
        
    def populate(self) -> np.array:
        # n = nodes, k = truss
        # system A matrix (n*3) x (k+2n)
        A = np.zeros((len(self._trusses) + len(self._nodes)*2, len(self._nodes)*3))
        n_rows, n_cols = A.shape
        n_nodes = len(self._nodes)
        n_trusses = len(self._trusses)

        # populate A matrix with displacement u,v
        # populate A matrix with force coefficient
        for k in range(0, n_trusses):
            truss = self._trusses[k]
            start, end = truss.get_nodes()
            alpha = truss.get_inclination()
            u = cos(alpha)
            v = sin(alpha)
            u1k, v1k = start.compute_displacement(u,v)
            i1 = start.get_index()
            u2k, v2k = end.compute_displacement(u,v)
            i2 = end.get_index()
            # Assign force coefficient to truss equation
            A[k,n_nodes*2+k] = truss.get_length()/truss._E/truss._A
            # Assign displacement to matrix, row = k, col = i
            print(u1k, v1k, u2k, v2k)
            A[k,i1] = u1k 
            A[k,i1+n_nodes] = v1k 
            A[k,i2] = u2k 
            A[k,i2+n_nodes] = v2k 
            # assign force coefficient in x,y node equation
            #A[n_trusses+i1, n_nodes*2+k] = cos(truss.get_inclination())
            #A[n_trusses+i1+1, n_nodes*2+k] = sin(truss.get_inclination())

        #print(A)
        return A