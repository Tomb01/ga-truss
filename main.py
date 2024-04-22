from src.classes.truss import *
from src.classes.contrains import *
from src.classes.structure import *
from src.utils.misc import make_brige_coordinate
import numpy as np
from math import isclose
from src.graphics.drawing import *

if __name__ == "__main__":
    
    # Appello 13/07/2018 - Tema A https://drive.google.com/file/d/13siogH0Rv4tQqtWkeGlki3VHwjq172c0/view?usp=sharing
    # Trave a triangolo equilatero con 4 aste sul bordo inferiore
    drawing = Drawing(cairo.SVGSurface("out/truss.svg", 600, 600))
    
    A = 1
    E = 1
        
    coord = make_brige_coordinate(1, 4)
    nodes = list(map(lambda point: TrussNode(point[0], point[1]), coord))
    print(coord)
    trusses = list()
    for i in range(0, len(nodes)-2):
        trusses.append(Truss(nodes[i], nodes[i+1], A, E))
        trusses.append(Truss(nodes[i], nodes[i+2], A, E))
        
    trusses.append(Truss(nodes[-1], nodes[-2], A, E))   
    structure = Structure(nodes, trusses)
        
    drawing.draw_structure(structure)
        

