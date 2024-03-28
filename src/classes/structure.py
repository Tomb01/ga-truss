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
        """
        Populate A system matrix with displacement u,v and force coefficient
        With:
        * n = number of nodes (1, 2, ...n)
        * k = number of truss (A, B, ...k)
        * N = force trasmitted from truss to node
        * u,v = x and y node displacement

        The column structure of A matrix:

        | u1, v1, u2, v2, ... un, vn, NA, NB, ... Nk |

        for k truss equation and then for n nodes equation (first x, then y direction)
        Final matrix size will be (k+2n) x (k+2n)
        """

        A = np.zeros((len(self._trusses) + len(self._nodes)*2, len(self._nodes)*2+len(self._trusses)))
        n_nodes = len(self._nodes)
        n_trusses = len(self._trusses)

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
            A[k,i1] = u1k 
            A[k,i1+n_nodes] = v1k 
            A[k,i2] = u2k 
            A[k,i2+n_nodes] = v2k 
            # assign force coefficient in x,y node equation
            node_eq_col = n_nodes*2+k
            A[n_trusses+i1, node_eq_col] = cos(truss.get_inclination())
            A[n_trusses+i1+n_nodes, node_eq_col] = sin(truss.get_inclination())
            A[n_trusses+i2, node_eq_col] = cos(truss.get_inclination())
            A[n_trusses+i2+n_nodes, node_eq_col] = sin(truss.get_inclination())

        return A