import unittest
from math import sqrt, cos, pi, sin
import sys
import numpy as np

sys.path.insert(0, '../src')

from src.classes.truss import TrussNode, Truss
from src.classes.contrains import *
from src.classes.structure import Structure

# Tema 15/06/2023 - A, es.1 
node1 = TrussNode(2.0, 0.0)
node2 = TrussNode(0.42264973, 0.0)
node3 = TrussNode(0.0, 1.0)
node4 = TrussNode(1.0, 1.0)

HingeConstrain(node1)
HingeConstrain(node2)
HingeConstrain(node3)
        
trusses = [Truss(node1, node4, 1.0, 1.0), Truss(node2, node4, 1.0, 1.0), Truss(node3, node4, 1.0, 1.0)]
nodes = [node1, node2, node3, node4]

A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1/1.154700538*cos(60*pi/180) + 1/2 + 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1/1.154700538*sin(60*pi/180) + 1/2],
])

structure = Structure(nodes, trusses)

class TestTruss(unittest.TestCase):

    def test_truss_inclination(self):
        node1 = TrussNode(0.42264973,0)
        node2 = TrussNode(1,1)
        HingeConstrain(node1)
        # Create a simple truss
        truss = Truss(node1, node2, 1, 1)
        self.assertAlmostEqual(cos(truss.get_inclination()), 0.5)
        self.assertAlmostEqual(truss.get_length(), 1.154700538)

    def test_structure_check(self):
        A_calc = structure.populate()
        print(A_calc)
        np.testing.assert_almost_equal(A_calc, A)
        