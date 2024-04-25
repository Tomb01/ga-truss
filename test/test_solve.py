import unittest
import math
import sys
import numpy as np

sys.path.insert(0, '../src')

from src.classes.truss import TrussNode, Truss
from src.classes.contrains import *
from src.classes.structure import Structure
from src.utils.misc import make_brige_coordinate

SIN_60 = math.sqrt(3)/2

class TestSolve(unittest.TestCase):

    def test_truss_solve_1(self):
        # Appello 11/07/2023 - A
        E = 1
        A = 1
        l1 = math.sqrt(3**2-1.5**2)
        nodes = [
            TrussNode(0,0),
            TrussNode(2,2),
            TrussNode(4,0),
            TrussNode(4+l1, 1.5),
            TrussNode(4+l1, 0),
            TrussNode(4+l1, -1.5)
        ]
        
        trusses = [
            Truss(nodes[0], nodes[1], A, E),
            Truss(nodes[1], nodes[2], A, E),
            Truss(nodes[2], nodes[3], A, E),
            Truss(nodes[2], nodes[4], A, E),
            Truss(nodes[2], nodes[5], A, E)
        ]
        
        Load(nodes[1], -math.sqrt(2), 0)
        HingeConstrain(nodes[0])
        HingeConstrain(nodes[5])
        HingeConstrain(nodes[4])
        HingeConstrain(nodes[3])
        
        structure = Structure(nodes, trusses)
        x = structure.solve()
        
        print(x[0:10])
        u = x[2]
        v = x[3+5]
        
        # Soluzioni u,v al nodo 3 
        np.testing.assert_almost_equal(u, -0.7991, 4)
        np.testing.assert_almost_equal(v, 4.2426, 4)
        
    def test_truss_solve_2(self):
        
        # Appello 16/06/2014
        E = 1
        A = 1
        nodes = [
            TrussNode(0, 0),
            TrussNode(1, 0),
            TrussNode(2, 0),
            TrussNode(1, 1),
            TrussNode(1+math.tan(math.radians(30)), 1)         
        ]
        
        trusses = [
            Truss(nodes[0], nodes[3], A, E),
            Truss(nodes[1], nodes[3], A, E),
            Truss(nodes[2], nodes[3], A, E),
            Truss(nodes[1], nodes[4], A, E),
            Truss(nodes[3], nodes[4], A, E)
        ]
        
        HingeConstrain(nodes[0])
        HingeConstrain(nodes[1])
        HingeConstrain(nodes[2])
        
        Load(nodes[4], 0, 8/3)
        
        structure = Structure(nodes, trusses)
        x = structure.solve()
        
        print(x)
        
        # test u e v
        np.testing.assert_almost_equal([[0, 0, 0, -2.177, -3.066]], np.transpose(x[0:5]), 3)
        np.testing.assert_almost_equal([[0, 0, 0, 0, 5.876]], np.transpose(x[5:10]), 3)