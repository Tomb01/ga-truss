from typing import List, Dict, Tuple, Callable
import numpy as np
from src.classes.truss import Truss, TrussNode
from math import cos, sin, degrees
from src.utils.calc import get_truss_x_direction, get_truss_y_direction, add_to_matrix
from src.utils.types import FixedNodeData
from collections import Counter

class Structure:

    _nodes: List[TrussNode]
    _trusses: List[Truss]
    _systemA: np.array
    _n_constrain: int = 0
    _A: np.array
    _b: np.array
    _checked: bool = False
    _n_constrain_nodes: int = 0
    _free_nodes: List[TrussNode]

    def __init__(self, nodes: List[TrussNode], trusses: List[Truss], n_constrain: int) -> None:
        self._nodes = nodes
        self._trusses = trusses
        self._n_constrain_nodes = n_constrain

        self.check()
        self.assing_index()
        self.populate()

    def check(self) -> bool:
        node_check: Dict[str, int] = {}
        found_trusses = []

        for i, truss in enumerate(self._trusses):
            node = truss.get_nodes()
            id1 = node[0].get_id()
            id2 = node[1].get_id()
            
            if id1+id2 in found_trusses or id2+id1 in found_trusses:
                del self._trusses[i]
                continue
            
            found_trusses.append(id1+id2)
            
            if id1 in node_check:
                node_check[id1] = node_check[id1] + 1
            else:
                node_check[id1] = 1
            
            if id2 in node_check:
                node_check[id2] = node_check[id2] + 1
            else:
                node_check[id2] = 1

        node_list: Dict[str, bool] = {}
        for node in self._nodes:
            node_list[node.get_id()] = node
            h,r = node.get_constrain()
            self._n_constrain = self._n_constrain + 1 if h != 0 else self._n_constrain
            self._n_constrain = self._n_constrain + 1 if r != 0 else self._n_constrain

        #print(node_check)

        if (node_list.keys() == node_check.keys()) and min(node_check.values()) > 1:
            self._checked = True
            return True
        else:
            total_nodes = list(node_check.keys()) +  list(node_list.keys())
            c = Counter(total_nodes)
            free_ids = [k for k, v in c.items() if v == 1]
            self._free_nodes = [node_list[x] for x in free_ids]
            self._checked = False
            self.fix()
            #raise ValueError("Structure not correct. Free nodes")
        
    def fix(self) -> None:
        if not self._checked:
            self._nodes = [x for x in self._nodes if x not in self._free_nodes]
        
    def get_DOF(self) -> int:
        if self._checked:
            return 2*len(self._nodes) - self._n_constrain - len(self._trusses)
        else:
            return 1
    
    def _iterate_truss(self, func: Callable[[Truss], any])-> List[any]:
        ret = map(func, self._trusses)
        return list(ret)
    
    def get_loads(self) -> np.array:
        return self._iterate_truss(lambda t: t.get_load())
    
    def get_stress(self) -> np.array:
        return self._iterate_truss(lambda t: t.get_stress())
        
    def assing_index(self) -> None:
        for index in range(0, len(self._nodes)):
            self._nodes[index-0].set_index(index)
        
    def populate(self) -> Tuple[np.array, np.array]:

        n_nodes = len(self._nodes)
        n_trusses = len(self._trusses)
        n_constrain = self._n_constrain
        m = 4*n_nodes
        # linear system (m, n) x (n, 1) = (m, 1)
        A = np.zeros((m, m))
        b = np.zeros((n_constrain+2*n_nodes, 1))

        for k in range(0, n_trusses):
            truss = self._trusses[k]
            k_truss = truss.area*truss.E/truss.get_length()
            node1, node2 = truss.get_nodes()
            alpha = truss.get_inclination()
            #print(alpha)
            i1 = node1.get_index()
            i2 = node2.get_index()
            
            # x direction -> cos, y direction -> sin
            # displacement = EA/l * sin/cos(truss angle) 
            u = k_truss*cos(alpha)
            v = k_truss*sin(alpha)
            #print("{k:n} - alpha={alpha:.3f} u={u:.3f} v={v:.3f}".format(u=cos(alpha), v=sin(alpha), alpha=degrees(alpha), k=k))
            #print("{k:n} - node {index:n}: u={u:.3f}, v={v:.3f}".format(index=i1, u=u, v=v, k=k))
            #print("{k:n} - node {index:n}: u={u:.3f}, v={v:.3f}".format(index=i2, u=-u, v=-v, k=k))
            #el = input()
            # Add calculated displacement, - is because the difference (u2-u1)cos(a) + (v2-v1)sin(a)
            eq = np.zeros(m)
            eq[i2] = u
            eq[i1] = -u
            eq[i2+n_nodes] = v
            eq[i1+n_nodes] = -v
            
            # Get load
            P1x, P1y = node1.get_load()
            P2x, P2y = node2.get_load()
            
            # Get constrain
            h1, r1 = node1.get_constrain()
            h2, r2 = node2.get_constrain()
            
            A[i1, 2*n_nodes + i1] = h1
            A[i2, 2*n_nodes + i2] = h2
            A[i1+n_nodes, 3*n_nodes + i1] = r1
            A[i2+n_nodes, 3*n_nodes + i2] = r2
            A[2*n_nodes+i1, i1] = h1
            A[2*n_nodes+i2, i2] = h2
            A[3*n_nodes+i1, i1+n_nodes] = r1
            A[3*n_nodes+i2, i2+n_nodes] = r2
            
            # Apply node 1 equation
            # Node 1 = internal
            x_component = abs(cos(alpha))
            y_component = abs(sin(alpha))
            xdir1 = get_truss_x_direction(node1, node2)
            ydir1 = get_truss_y_direction(node1, node2)
            A[i1] = A[i1] + eq * x_component * xdir1
            A[i1+n_nodes] = A[i1+n_nodes] + eq * y_component * ydir1
            b[i1] = P1x
            b[i1+n_nodes] = P1y
            
            # Apply node 2 equation
            # Node 2 = internal
            xdir2 = get_truss_x_direction(node2, node1) 
            ydir2 = get_truss_y_direction(node2, node1)
            A[i2] = A[i2] + eq * x_component * xdir2
            A[i2+n_nodes] = A[i2+n_nodes] + eq * y_component * ydir2
            b[i2] = P2x
            b[i2+n_nodes] = P2y
        
        #Resize matrix to delete all zeros row and col
        #https://stackoverflow.com/a/51770413
        
        idx = np.argwhere(np.all(A[..., :] == 0, axis=0))
        #print(idx)
        A = np.delete(A, idx, axis=1)
        #print(~np.all(A == 0, axis=1))
        A = A[~np.all(A == 0, axis=1)]
        
        self._A = A
        self._b = b
        
        return A, b   
    
    def solve(self) -> np.array:
        n_nodes = len(self._nodes)
        x = np.linalg.solve(self._A, self._b)
        x_t = x.transpose()[0]
        #print(x_t)
        for node in self._nodes:
            index = node.get_index()
            node.set_displacement(x_t[index], x_t[index+n_nodes]) 
            
        for truss in self._trusses:
            truss.calculate()       
        return x
    
    
def get_bound_box(fixed: List[TrussNode]) -> Tuple[float, float, float, float]:
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    for p in fixed:
        x,y = p.get_coordinate()
        max_x = x if x > max_x else max_x
        max_y = y if y > max_y else max_y
        min_x = x if x < min_x else min_x
        min_y = y if y < min_y else min_y
        
    return (min_x, min_y, max_x, max_y)
        