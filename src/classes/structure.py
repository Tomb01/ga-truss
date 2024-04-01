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

        n_nodes = len(self._nodes)
        n_trusses = len(self._trusses)
        A = np.zeros((n_nodes*2, n_nodes*2))

        for k in range(0, n_trusses):
            truss = self._trusses[k]
            start, end = truss.get_nodes()
            alpha = truss.get_inclination()
            u = cos(alpha)
            v = sin(alpha)
            u1, v1 = start.compute_displacement(u, v)
            u2, v2 = end.compute_displacement(u, v)
            i1 = start.get_index()
            i2 = end.get_index()
            #print(u1, u2, v1, v2)
            tk = truss.area*truss.E/truss.get_length()
            Nu1 = tk*u1
            Nu2 = tk*u2
            Nv1 = tk*v1
            Nv2 = tk*v2
            
            # Node 1 equation, x direction
            A[i1, i1] = Nu1*cos(alpha)
            A[i1, i2] = Nu2*cos(alpha)
            A[i1, i1+n_nodes] = Nv1*cos(alpha)
            A[i1, i2+n_nodes] = Nv2*cos(alpha)
            # Node 1 equation, y direction
            A[i1+n_nodes, i1] = Nu1*sin(alpha)
            A[i1+n_nodes, i2] = Nu2*sin(alpha)
            A[i1+n_nodes, i1+n_nodes] = Nv1*sin(alpha)
            A[i1+n_nodes, i2+n_nodes] = Nv2*sin(alpha)
            
            # Node 2 equation, x direction
            A[i2, i1] = Nu1*cos(alpha)
            A[i2, i2] = Nu2*cos(alpha)
            A[i2, i1+n_nodes] = Nv1*cos(alpha)
            A[i2, i2+n_nodes] = Nv2*cos(alpha)
            # Node 2 equation, y direction
            A[i2+n_nodes, i1] = Nu1*sin(alpha)
            A[i2+n_nodes, i2] = Nu2*sin(alpha)
            A[i2+n_nodes, i1+n_nodes] = Nv1*sin(alpha)
            A[i2+n_nodes, i2+n_nodes] = Nv2*sin(alpha)
            
        return A