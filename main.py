from src.classes.playground import PlayGround
from src.classes.material import Material
import src.graphics.plot as plt

if __name__ == "__main__":
    
    material = Material()
    material.E = 210 * 100 #Gpa -> N/mm^2
    material.Re = 600 # N/mm^2
    
    area = 2 #mm^2
    
    playground = PlayGround(max_epochs=100, population=100)
    playground.play([
        [0, 0, True, True, 0, 1*1e6],
        [1000, 1000, True, True, 0, 1*1e6]
    ], material, 1, area)
    
    #plt.plot_structure(phenotype._structure)
    #plt.show()