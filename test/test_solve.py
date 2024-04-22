import unittest
from math import sqrt, cos, pi, sin
import sys
import numpy as np

sys.path.insert(0, '../src')

from src.classes.truss import TrussNode, Truss
from src.classes.contrains import *
from src.classes.structure import Structure
from src.utils.misc import make_brige_coordinate

SIN_60 = sqrt(3)/2

class TestSolve(unittest.TestCase):

    def test_truss_solve_1(self):
        # Appello 13/07/2018 - Tema A https://drive.google.com/file/d/13siogH0Rv4tQqtWkeGlki3VHwjq172c0/view?usp=sharing
        # Trave a triangolo equilatero con 4 aste sul bordo inferiore
        A = 1
        E = 1
        
        coord = make_brige_coordinate(1, 4)
        nodes = map(coord, lambda x,y: TrussNode(x, y))
        truss = list()
        for i in range(0, len(nodes)-1):
            truss.append(Truss(nodes[i], nodes[i+1], A, E))
            truss.append(Truss(nodes[i], nodes[i+2], A, E))
        
        print(truss)