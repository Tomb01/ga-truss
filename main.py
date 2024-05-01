from src.classes.genome import Phenotype
from src.classes.material import Material
#from src.graphics.drawing import *

if __name__ == "__main__":
    
    material = Material()
    material.E = 1
    
    phenotype = Phenotype.random([
        [0, 0, True, True, 0, 1000],
        [10, 10, True, True, 0, 1000]
    ], material, 5)
    
    