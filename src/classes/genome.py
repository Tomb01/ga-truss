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
    def random(cls, geometric_constrains: List[FixedNodeData], material: Material, area = 10, node_n: int = 10) -> 'Phenotype':
        nodes = list(map(lambda p: PointConstrain(p[0], p[1], p[2], p[3], p[4], p[5]), geometric_constrains))
        x_min, y_min, x_max, y_max = get_bound_box(nodes)
        x_rand = np.random.uniform(low=x_min, high=x_max, size=(node_n,))
        y_rand = np.random.uniform(low=y_min, high=y_max, size=(node_n,))
        print(x_rand, y_rand)
        for i in range(0, len(x_rand)):
            nodes.append(TrussNode(x_rand[i], y_rand[i]))
        
        r = 0 # Number of constrains (displacement equal to zero)
        for i in range(0, len(geometric_constrains)):
            if geometric_constrains[i][2]:
                r = r+1
            if geometric_constrains[i][3]:
                r = r+1
                
        j = len(x_rand) # Number of nodes
        trusses = list()
        m = 2*j - r # Number of truss required for an isostatic structure
        for i in range(0, len(nodes)):
            k = random.randrange(len(nodes))
            if i==k:
                k = k + 1
            trusses.append(Truss(nodes[i], nodes[k], area, material.E))
            
        structure = Structure(nodes, trusses)
        print(nodes)
        
        return cls(structure, material)
    
    _structure: Structure
    _material: Material
    
    def __init__(self, structure: Structure, material: Material):
        self._structure = structure
        self._material = material
        
    def get_score(self) -> float:
        """
        Function to score fit calculation
        """
        Fos = self._material.Re / self._structure.get_stress() # Fos = yield stress/working stres
                
        
def fit_function(mass: float, Fos: np.array) -> float:
    return 0