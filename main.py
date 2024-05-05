from src.classes.genome import Phenotype
from src.classes.material import Material
import src.graphics.plot as plt

if __name__ == "__main__":
    
    material = Material()
    material.E = 1
    
    phenotype = Phenotype.random([
        [0, 0, True, True, 0, 1000],
        [100, 100, True, True, 0, 1000]
    ], material, 5)
    
    plt.plot_structure(phenotype._structure)
    plt.show()