from src.classes.structure import Structure, get_bound_box
from src.classes.contrains import Constrain, Load
from src.classes.node import TrussNode
from src.classes.truss import Truss
from src.classes.material import Material
from typing import List
from src.utils.types import FixedNodeData
import random
import numpy as np

def PointConstrain(x: float, y: float, x_lock: bool = True, y_lock: bool = True, x_load: float = 0, y_load: float = 0):
    node = TrussNode(x, y)
    Constrain(lambda : (1 if x_lock else 0, 1 if y_lock else 0), node)
    Load(node, x_load, y_load)
    return node

class Phenotype:
    
    @classmethod
    def random(cls, geometric_constrains: List[FixedNodeData], material: Material, area = 10, truss_n: int = 10) -> 'Phenotype':
        nodes = list(map(lambda p: PointConstrain(p[0], p[1], p[2], p[3], p[4], p[5]), geometric_constrains))
        dof = truss_n*3 # Degree of freedom
        
        v_e = 0 # Number of external constrains (displacement equal to zero)
        for i in range(0, len(geometric_constrains)):
            if geometric_constrains[i][2]:
                v_e = v_e+1
            if geometric_constrains[i][3]:
                v_e = v_e+1
                
        n_nodes = int((dof - v_e)/2) # Minimum nodes required for isostatic structure
        
        x_min, y_min, x_max, y_max = get_bound_box(nodes)
        x_rand = np.random.uniform(low=x_min, high=x_max, size=(n_nodes,))
        y_rand = np.random.uniform(low=y_min, high=y_max, size=(n_nodes,))
        for i in range(0, len(x_rand)):
            nodes.append(TrussNode(x_rand[i], y_rand[i]))
            
        trusses = list()
        truss_node_match = [ [] for _ in range(len(nodes)) ]
        #print(x_rand, y_rand)
                
        for i in range(0, len(nodes)):
            k = random.randrange(len(nodes))
            if i==k:
                k = k+1

            if k < len(truss_node_match):
                truss_node_match[k].append(i)
                trusses.append(Truss(nodes[i], nodes[k], area, material.E))
            
        free_nodes = list()
        for i,x in enumerate(truss_node_match):
            if len(x) <= 1:
                free_nodes.append(i)
        #print(free_nodes)        
        
        for i in range(0, len(nodes)):
            k = random.choice(free_nodes)
            if i == k or i in truss_node_match[k]:
                continue
            free_nodes.remove(k)
            truss_node_match[k].append(i)
            #print(i, k, truss_node_match)
            trusses.append(Truss(nodes[i], nodes[k], area, material.E))
            if len(free_nodes)==0:
                break
        
        structure = Structure(nodes, trusses)
        #print(truss_node_match)
        
        return cls(structure, material)
    
    _structure: Structure
    _material: Material
    
    def __init__(self, structure: Structure, material: Material):
        self._structure = structure
        self._material = material
        
    def get_score(self, better_Fos = 2) -> float:
        """
        Function to score fit calculation
        """
    
        k_DOF = 1
        k_Fos = 1
        k_Mass = 1
        
        ########## DOF objective ##########
        # Se i DOF sono maggiori di 0 la struttura è labile -> score = 0
        # Se la matrice è singolare la struttura non ha soluzione
        try:
            self._structure.solve()
            k_DOF = 1 if self._structure.get_DOF() <= 0 else 0
        except np.linalg.LinAlgError as e:
            if e == "Singular matrix":
                k_DOF = 0
                print("Singular matrix " + str(self._structure.get_DOF()))

        ########## FoS objective ##########
        # Ottimizzazione del valore di Fos per ogni trave.
        # Con Fos < 1 la struttura si rompe -> score = 0
        Fos = np.nan_to_num(abs(np.divide(self._material.Re, self._structure.get_stress()))) # Fos = yield stress/working stres
        is_broken = np.any(Fos < 1)
        if is_broken:
            k_Fos = 0
        else:
            Fos_margin = Fos - better_Fos
            Fos_margin_mean = np.mean(Fos_margin)
            
            k_Fos = 1/Fos_margin_mean
        
        return k_DOF * k_Fos * k_Mass
                
    