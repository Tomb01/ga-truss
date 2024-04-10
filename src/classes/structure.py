from typing import List, Dict, Tuple
import numpy as np
from src.classes.truss import Truss, TrussNode
from math import cos, sin
from src.utils.calc import get_truss_x_direction, get_truss_y_direction, add_to_matrix

class Structure:

    _nodes: List[TrussNode]
    _trusses: List[Truss]
    _systemA: np.array
    _n_constrain: int = 0

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
            if node.is_constrained_x():
                self._n_constrain = self._n_constrain+1
            if node.is_constrained_y():
                self._n_constrain = self._n_constrain+1

        #print(node_check, node_list)

        if node_list == node_check:
            return True
        else:
            return False
        
    def get_DOF(self) -> int:
        raise NotImplementedError
        
    def assing_index(self) -> None:
        for index in range(0, len(self._nodes)):
            self._nodes[index-0].set_index(index)
        
    def populate(self) -> Tuple[np.array, np.array]:

        n_nodes = len(self._nodes)
        n_trusses = len(self._trusses)
        m = n_nodes*2 + self._n_constrain
        # linear system (m, n) x (n, 1) = (m, 1)
        A = np.zeros((m, m))
        b = np.zeros((m, 1))
        constrain_i: int = 0

        for k in range(0, n_trusses):
            truss = self._trusses[k]
            k_truss = truss.area*truss.E/truss.get_length()
            node1, node2 = truss.get_nodes()
            alpha = truss.get_inclination()
            i1 = node1.get_index()
            i2 = node2.get_index()
            
            # x direction -> cos, y direction -> sin
            # displacement = sin/cos(force angle) * EA/l * sin/cos(truss angle) 
            u = cos(alpha)*k_truss*cos(alpha)
            v = sin(alpha)*k_truss*sin(alpha)
            print("{k:n} - node {index:n}: u={u:.3f}, v={v:.3f}".format(index=i1, u=u, v=v, k=k))
            print("{k:n} - node {index:n}: u={u:.3f}, v={v:.3f}".format(index=i2, u=-u, v=-v, k=k))
            # Add calculated displacement, - is because the difference (u2-u1)cos(a) + (v2-v1)sin(a)
            eq = np.zeros(m)
            eq[i2] = u
            eq[i1] = -u
            eq[i2+n_nodes] = v
            eq[i1+n_nodes] = -v
            
            # Apply node 1 equation
            # Node 1 = internal
            xdir1 = get_truss_x_direction(node1, node2)
            ydir1 = get_truss_y_direction(node1, node2)
            A[i1] = A[i1] + eq * xdir1
            A[i1+n_nodes] = A[i1+n_nodes] + eq * ydir1
            
            # Apply node 2 equation
            # Node 2 = internal
            xdir2 = get_truss_x_direction(node2, node1)
            ydir2 = get_truss_y_direction(node2, node1)
            A[i2] = A[i2] + eq*xdir2
            A[i2+n_nodes] = A[i2+n_nodes] + eq*ydir2
        
        return A, b   